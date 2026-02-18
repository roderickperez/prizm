from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import segyio
except ImportError:
    segyio = None


class SegyDataStore:
    def __init__(self, segy_path: Path, logger=None, cache_size: int = 12):
        self.segy_path = Path(segy_path)
        self.logger = logger
        self.cache_size = cache_size

        self._segy_file = None
        self._slice_cache: OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray, tuple[str, str]]] = OrderedDict()

        self.has_data = False
        self.sample_count = 0
        self.amp_limit = 1000.0
        self.dims = (0, 0, 0)

        self.inline_values = np.array([], dtype=int)
        self.xline_values = np.array([], dtype=int)
        self.timeslice_values = np.array([], dtype=int)

        self.inline_index_map: dict[int, int] = {}
        self.xline_index_map: dict[int, int] = {}
        self.timeslice_index_map: dict[int, int] = {}

        self.trace_index_grid: Optional[np.ndarray] = None
        self.valid_trace_positions: Optional[np.ndarray] = None

    def _log(self, level: str, message: str) -> None:
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if callable(log_fn):
            log_fn(message)

    def _open_file(self):
        if self._segy_file is None:
            self._segy_file = segyio.open(str(self.segy_path), mode="r", ignore_geometry=True)
            self._segy_file.mmap()
        return self._segy_file

    def load(self) -> bool:
        if segyio is None:
            self._log("warning", "segyio is not installed. Install it to load local seismic data.")
            return False
        if not self.segy_path.exists():
            self._log("warning", f"SEGY file not found: {self.segy_path}")
            return False

        try:
            segy_file = self._open_file()
            trace_count = segy_file.tracecount
            sample_values = np.asarray(getattr(segy_file, "samples", []))
            self.sample_count = int(len(sample_values))

            inline_headers = np.asarray(
                segy_file.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int32
            )
            xline_headers = np.asarray(
                segy_file.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int32
            )

            if inline_headers.size != trace_count:
                inline_headers = np.array([int(segy_file.header[i].get(segyio.TraceField.INLINE_3D, i)) for i in range(trace_count)], dtype=np.int32)
            if xline_headers.size != trace_count:
                xline_headers = np.array([int(segy_file.header[i].get(segyio.TraceField.CROSSLINE_3D, 0)) for i in range(trace_count)], dtype=np.int32)

            inline_values = np.unique(inline_headers)
            xline_values = np.unique(xline_headers)
            if inline_values.size == 0 or xline_values.size == 0 or self.sample_count <= 0:
                raise ValueError("Unable to derive inline/xline/sample axes from SEG-Y headers")

            inline_map = {int(v): i for i, v in enumerate(inline_values)}
            xline_map = {int(v): i for i, v in enumerate(xline_values)}
            trace_index_grid = np.full((inline_values.size, xline_values.size), -1, dtype=np.int64)

            for trace_idx in range(trace_count):
                il_pos = inline_map.get(int(inline_headers[trace_idx]))
                xl_pos = xline_map.get(int(xline_headers[trace_idx]))
                if il_pos is None or xl_pos is None:
                    continue
                if trace_index_grid[il_pos, xl_pos] < 0:
                    trace_index_grid[il_pos, xl_pos] = trace_idx

            amp_step = max(1, trace_count // 2000)
            amp_samples = []
            for trace_idx in range(0, trace_count, amp_step):
                trace = np.asarray(segy_file.trace[trace_idx], dtype=np.float32)
                amp_samples.append(np.abs(trace))
            if amp_samples:
                amp_concat = np.concatenate(amp_samples)
                self.amp_limit = float(np.percentile(amp_concat, 99))

            if sample_values.size != self.sample_count:
                timeslice_values = np.arange(self.sample_count, dtype=int)
            else:
                timeslice_values = np.round(sample_values).astype(int)
                if np.unique(timeslice_values).size != self.sample_count:
                    timeslice_values = np.arange(self.sample_count, dtype=int)

            self.inline_values = inline_values.astype(int)
            self.xline_values = xline_values.astype(int)
            self.timeslice_values = timeslice_values.astype(int)
            self.inline_index_map = {int(v): i for i, v in enumerate(self.inline_values)}
            self.xline_index_map = {int(v): i for i, v in enumerate(self.xline_values)}
            self.timeslice_index_map = {int(v): i for i, v in enumerate(self.timeslice_values)}
            self.trace_index_grid = trace_index_grid
            self.valid_trace_positions = np.argwhere(trace_index_grid >= 0)
            self.dims = (self.inline_values.size, self.xline_values.size, self.timeslice_values.size)

            self.has_data = True
            self._log("info", f"Loaded SEG-Y index grid: {self.dims} from {self.segy_path}")
            return True
        except Exception as exc:
            self.has_data = False
            self.trace_index_grid = None
            self.valid_trace_positions = None
            self._log("exception", f"SEGY load error: {exc}")
            return False

    def _cache_get(self, key: tuple[str, int]):
        if key in self._slice_cache:
            value = self._slice_cache.pop(key)
            self._slice_cache[key] = value
            return value
        return None

    def _cache_set(self, key: tuple[str, int], value):
        if key in self._slice_cache:
            self._slice_cache.pop(key)
        self._slice_cache[key] = value
        while len(self._slice_cache) > self.cache_size:
            self._slice_cache.popitem(last=False)

    def get_slice(self, mode: str, idx: int):
        if not self.has_data or self.trace_index_grid is None:
            raise RuntimeError("SEG-Y trace index grid is unavailable")

        key = (mode, int(idx))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        i_ext, j_ext, k_ext = self.dims
        segy_file = self._open_file()

        if mode == "Inline":
            fallback = i_ext // 2
            il_idx = int(np.clip(self.inline_index_map.get(int(idx), fallback), 0, i_ext - 1))
            trace_ids = self.trace_index_grid[il_idx, :]
            data = np.full((j_ext, k_ext), np.nan, dtype=np.float32)
            valid = np.where(trace_ids >= 0)[0]
            for j_pos in valid:
                data[j_pos, :] = np.asarray(segy_file.trace[int(trace_ids[j_pos])], dtype=np.float32)
            result = (data, data, ("Crossline", "Time/Depth"))
            self._cache_set(key, result)
            return result

        if mode == "Crossline":
            fallback = j_ext // 2
            xl_idx = int(np.clip(self.xline_index_map.get(int(idx), fallback), 0, j_ext - 1))
            trace_ids = self.trace_index_grid[:, xl_idx]
            data = np.full((i_ext, k_ext), np.nan, dtype=np.float32)
            valid = np.where(trace_ids >= 0)[0]
            for i_pos in valid:
                data[i_pos, :] = np.asarray(segy_file.trace[int(trace_ids[i_pos])], dtype=np.float32)
            result = (data, data, ("Inline", "Time/Depth"))
            self._cache_set(key, result)
            return result

        fallback = k_ext // 2
        sample_idx = int(np.clip(self.timeslice_index_map.get(int(idx), fallback), 0, k_ext - 1))
        pad = 16
        z_start = max(0, sample_idx - pad)
        z_end = min(k_ext - 1, sample_idx + pad)
        z_len = z_end - z_start + 1
        chunk = np.full((i_ext, j_ext, z_len), np.nan, dtype=np.float32)

        valid_positions = self.valid_trace_positions
        if valid_positions is None:
            valid_positions = np.argwhere(self.trace_index_grid >= 0)
            self.valid_trace_positions = valid_positions
        for il_pos, xl_pos in valid_positions:
            trace_idx = int(self.trace_index_grid[il_pos, xl_pos])
            trace = np.asarray(segy_file.trace[trace_idx], dtype=np.float32)
            chunk[il_pos, xl_pos, :] = trace[z_start : z_end + 1]

        rel_idx = sample_idx - z_start
        result = (chunk, chunk[:, :, rel_idx], ("Inline", "Crossline"))
        self._cache_set(key, result)
        return result
