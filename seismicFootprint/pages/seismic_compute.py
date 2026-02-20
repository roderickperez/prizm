from __future__ import annotations

import concurrent.futures
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import segyio
except ImportError:
    segyio = None

try:
    from PyEMD import EEMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False


ProgressCallback = Optional[Callable[[int, int], None]]


def _emd_worker(signal: np.ndarray, imfs_to_remove: int, trials: int, noise_width: float) -> tuple[np.ndarray, np.ndarray]:
    if not HAS_EMD:
        return signal, np.zeros_like(signal)

    if np.all(np.isnan(signal)):
        return signal, np.zeros_like(signal)

    try:
        eemd = EEMD(trials=int(max(1, trials)), noise_width=float(max(0.0, noise_width)))
        imfs = eemd.eemd(signal)
        if imfs is None or imfs.size == 0:
            return signal, np.zeros_like(signal)

        start_idx = int(max(0, imfs_to_remove))
        if imfs.shape[0] <= start_idx:
            return signal, np.zeros_like(signal)

        denoised = np.sum(imfs[start_idx:], axis=0)
        noise = signal - denoised
        return denoised.astype(np.float32), noise.astype(np.float32)
    except Exception:
        return signal, np.zeros_like(signal)


class FootprintComputeEngine:
    def __init__(self, logger=None, cache_size: int = 16):
        self.logger = logger
        self.cache_size = int(max(4, cache_size))

        self._segy_file = None
        self._slice_cache: OrderedDict[tuple[str, int], tuple[np.ndarray, tuple[str, str]]] = OrderedDict()

        self.has_test_volume = False
        self.dims = (0, 0, 0)
        self.amp_limit = 1000.0

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
        fn = getattr(self.logger, level, None)
        if callable(fn):
            fn(message)

    def _cache_get(self, key: tuple[str, int]):
        if key not in self._slice_cache:
            return None
        value = self._slice_cache.pop(key)
        self._slice_cache[key] = value
        return value

    def _cache_set(self, key: tuple[str, int], value) -> None:
        if key in self._slice_cache:
            self._slice_cache.pop(key)
        self._slice_cache[key] = value
        while len(self._slice_cache) > self.cache_size:
            self._slice_cache.popitem(last=False)

    def _open_segy(self, segy_path: Path):
        if self._segy_file is None:
            self._segy_file = segyio.open(str(segy_path), mode="r", ignore_geometry=True)
            self._segy_file.mmap()
        return self._segy_file

    def load_test_volume(self, segy_path: Path) -> bool:
        segy_path = Path(segy_path)
        if segyio is None:
            self._log("warning", "segyio is not installed; cannot load test SEG-Y volume")
            return False
        if not segy_path.exists():
            self._log("warning", f"SEG-Y file not found: {segy_path}")
            return False

        try:
            segy_file = self._open_segy(segy_path)
            trace_count = int(segy_file.tracecount)
            sample_values = np.asarray(getattr(segy_file, "samples", []))
            sample_count = int(len(sample_values))

            inline_headers = np.asarray(segy_file.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int32)
            xline_headers = np.asarray(segy_file.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int32)

            if inline_headers.size != trace_count:
                inline_headers = np.array(
                    [int(segy_file.header[i].get(segyio.TraceField.INLINE_3D, i)) for i in range(trace_count)], dtype=np.int32
                )
            if xline_headers.size != trace_count:
                xline_headers = np.array(
                    [int(segy_file.header[i].get(segyio.TraceField.CROSSLINE_3D, 0)) for i in range(trace_count)], dtype=np.int32
                )

            inline_values = np.unique(inline_headers)
            xline_values = np.unique(xline_headers)
            if inline_values.size == 0 or xline_values.size == 0 or sample_count <= 0:
                raise ValueError("Could not derive inline/xline/sample axes from SEG-Y")

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

            amp_step = max(1, trace_count // 1500)
            amp_samples = []
            for trace_idx in range(0, trace_count, amp_step):
                trace = np.asarray(segy_file.trace[trace_idx], dtype=np.float32)
                amp_samples.append(np.abs(trace))
            if amp_samples:
                self.amp_limit = float(np.percentile(np.concatenate(amp_samples), 99))

            if sample_values.size != sample_count:
                timeslice_values = np.arange(sample_count, dtype=int)
            else:
                timeslice_values = np.round(sample_values).astype(int)
                if np.unique(timeslice_values).size != sample_count:
                    timeslice_values = np.arange(sample_count, dtype=int)

            self.inline_values = inline_values.astype(int)
            self.xline_values = xline_values.astype(int)
            self.timeslice_values = timeslice_values.astype(int)

            self.inline_index_map = {int(v): i for i, v in enumerate(self.inline_values)}
            self.xline_index_map = {int(v): i for i, v in enumerate(self.xline_values)}
            self.timeslice_index_map = {int(v): i for i, v in enumerate(self.timeslice_values)}

            self.trace_index_grid = trace_index_grid
            self.valid_trace_positions = np.argwhere(trace_index_grid >= 0)
            self.dims = (self.inline_values.size, self.xline_values.size, self.timeslice_values.size)
            self.has_test_volume = True
            self._log("info", f"Loaded test SEG-Y with dims {self.dims}")
            return True
        except Exception as exc:
            self._log("exception", f"Failed to load SEG-Y: {exc}")
            self.has_test_volume = False
            self.trace_index_grid = None
            self.valid_trace_positions = None
            return False

    def get_test_slice(self, mode: str, idx: int) -> tuple[np.ndarray, tuple[str, str]]:
        if not self.has_test_volume or self.trace_index_grid is None:
            raise RuntimeError("Test SEG-Y volume is not loaded")

        key = (mode, int(idx))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        i_ext, j_ext, k_ext = self.dims
        segy_file = self._segy_file
        if segy_file is None:
            raise RuntimeError("SEG-Y file handle is not initialized")

        if mode == "Inline":
            fallback = i_ext // 2
            il_idx = int(np.clip(self.inline_index_map.get(int(idx), fallback), 0, i_ext - 1))
            trace_ids = self.trace_index_grid[il_idx, :]
            data = np.full((j_ext, k_ext), np.nan, dtype=np.float32)
            valid = np.where(trace_ids >= 0)[0]
            for j_pos in valid:
                data[j_pos, :] = np.asarray(segy_file.trace[int(trace_ids[j_pos])], dtype=np.float32)
            result = (data, ("Crossline", "Time/Depth"))
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
            result = (data, ("Inline", "Time/Depth"))
            self._cache_set(key, result)
            return result

        fallback = k_ext // 2
        sample_idx = int(np.clip(self.timeslice_index_map.get(int(idx), fallback), 0, k_ext - 1))
        data = np.full((i_ext, j_ext), np.nan, dtype=np.float32)

        valid_positions = self.valid_trace_positions
        if valid_positions is None:
            valid_positions = np.argwhere(self.trace_index_grid >= 0)
            self.valid_trace_positions = valid_positions

        for il_pos, xl_pos in valid_positions:
            trace_idx = int(self.trace_index_grid[il_pos, xl_pos])
            trace = np.asarray(segy_file.trace[trace_idx], dtype=np.float32)
            data[il_pos, xl_pos] = trace[sample_idx]

        result = (data, ("Inline", "Crossline"))
        self._cache_set(key, result)
        return result

    def apply_eemd_parallel(
        self,
        data_2d: np.ndarray,
        axis: int,
        imfs_to_remove: int,
        trials: int = 12,
        noise_width: float = 0.05,
        max_workers: Optional[int] = None,
        progress_callback: ProgressCallback = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if data_2d is None:
            raise ValueError("Input data_2d is None")

        data_2d = np.asarray(data_2d, dtype=np.float32)
        if data_2d.ndim != 2:
            raise ValueError("apply_eemd_parallel expects a 2D array")

        if not HAS_EMD:
            return data_2d.copy(), np.zeros_like(data_2d)

        rows, cols = data_2d.shape
        denoised = np.zeros_like(data_2d)
        noise = np.zeros_like(data_2d)

        axis = 1 if int(axis) == 1 else 0
        tasks = [data_2d[r, :] for r in range(rows)] if axis == 1 else [data_2d[:, c] for c in range(cols)]
        total_tasks = len(tasks)

        if total_tasks == 0:
            return data_2d.copy(), np.zeros_like(data_2d)

        results = []
        worker_count = max_workers if max_workers is not None else min(8, total_tasks)

        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_emd_worker, signal, int(imfs_to_remove), int(trials), float(noise_width)): idx
                for idx, signal in enumerate(tasks)
            }

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    clean, dirty = future.result(timeout=90)
                except Exception:
                    clean, dirty = tasks[idx], np.zeros_like(tasks[idx])
                results.append((idx, clean, dirty))
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total_tasks)

        for idx, clean, dirty in results:
            if axis == 1:
                denoised[idx, :] = clean
                noise[idx, :] = dirty
            else:
                denoised[:, idx] = clean
                noise[:, idx] = dirty

        return denoised, noise
