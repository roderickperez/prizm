import logging
import time
from pathlib import Path


def setup_app_logger(log_dir: Path, logger_name: str = "seismic_attributes_app") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"seismic_attributes_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    existing_logs = sorted(log_dir.glob("seismic_attributes_*.log"), key=lambda p: p.stat().st_mtime)
    while len(existing_logs) > 10:
        oldest = existing_logs.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass

    logger.info("Logger initialized")
    return logger
