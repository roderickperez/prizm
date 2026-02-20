from __future__ import annotations

from pathlib import Path

import numpy as np

from petrel_surface_parser import parse_petrel_surface_file


PETREL_FILE = Path(
    "/home/roderickperez/DataScienceProjects/Prizm_OMV/referenceDocumentation/structuralUncertaintyEvaluation/testData/surfaceTest"
)


def build_negative_synthetic_examples(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, np.ndarray]:
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    x0 = float(np.mean(x_values))
    y0 = float(np.mean(y_values))

    twt_synthetic = -1850.0 + 260.0 * np.exp(-(((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2.0 * 2000.0**2)))
    vel_synthetic = 3050.0 + 100.0 * np.exp(-(((x_grid - (x0 + 1200.0)) ** 2 + (y_grid - (y0 - 700.0)) ** 2) / (2.0 * 3200.0**2)))

    theta = np.deg2rad(30.0)
    xr = (x_grid - x0) * np.cos(theta) + (y_grid - y0) * np.sin(theta)
    yr = -(x_grid - x0) * np.sin(theta) + (y_grid - y0) * np.cos(theta)
    twt_elongated = -1900.0 + 320.0 * np.exp(-((xr**2) / (2.0 * 2800.0**2) + (yr**2) / (2.0 * 1100.0**2)))
    vel_elongated = 3100.0 + 120.0 * np.exp(-(((x_grid - x0) ** 2) / (2.0 * 3300.0**2) + ((y_grid - y0) ** 2) / (2.0 * 1500.0**2)))

    return {
        "twt_synthetic": twt_synthetic.astype(np.float32),
        "vel_synthetic": vel_synthetic.astype(np.float32),
        "twt_elongated": twt_elongated.astype(np.float32),
        "vel_elongated": vel_elongated.astype(np.float32),
    }


def print_range(name: str, data: np.ndarray) -> None:
    print(f"{name}: min={float(np.min(data)):.3f}, max={float(np.max(data)):.3f}")


if __name__ == "__main__":
    parsed = parse_petrel_surface_file(PETREL_FILE)
    x = np.asarray(parsed["x"], dtype=np.float32)
    y = np.asarray(parsed["y"], dtype=np.float32)
    z = np.asarray(parsed["z"], dtype=np.float32)
    metadata = parsed["metadata"]

    print("Petrel Surface Parsed")
    print(f"file: {PETREL_FILE}")
    print(f"grid(header): {metadata['grid_size_header']}")
    print(f"grid(detected): {metadata['grid_size_detected']}")
    print_range("X", x)
    print_range("Y", y)
    print_range("Z (imported surface)", z)

    synth = build_negative_synthetic_examples(x, y)
    print("\nSynthetic Option 1 (Negative TWT)")
    print_range("TWT", synth["twt_synthetic"])
    print_range("Velocity", synth["vel_synthetic"])

    print("\nSynthetic Option 2 (Negative TWT Elongated)")
    print_range("TWT", synth["twt_elongated"])
    print_range("Velocity", synth["vel_elongated"])

    print("\nImported Option 3 (Petrel Test Surface)")
    print_range("TWT/Depth", z)