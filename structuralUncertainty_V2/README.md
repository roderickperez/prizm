# Structural Uncertainty Module — Detailed Developer README

This document explains the Structural Uncertainty module end-to-end:

- architecture
- workflow logic
- formulas and units
- UI controls and update triggers
- plotting pipeline
- logging and retention
- launcher/constants integration
- how to run and troubleshoot

Latest implementation updates in this version:

- compute logic split into reusable core module (`core/engine_V2.py`)
- optional accelerated realization path (Torch when available)
- progress bar for realization + trap-statistics Monte Carlo processing
- new surface mode: `Elongated / Ellipsoidal` with conditional geometry sliders
- map cut-line angle control (0–360°) with perpendicular structural section extraction
- V2 wells workflow with compact table-based anchor-point editing (`Well Name`, `X`, `Y`, `Velocity`)
- multi-well covariance conditioning in the realization engine (`core/engine_V2.py`)
- report/QC expansion with well diagnostics, covariance exports, empirical/theoretical variogram diagnostics, engineering coherence checks, and recommended variogram parameter outputs
- explicit apply workflow for section controls (surface/wells/general settings now require their apply button)

---

## 1) Module Purpose

The module evaluates depth-conversion uncertainty by following a geostatistical workflow:

1. Start from a deterministic TWT surface (input map).
2. Generate stochastic velocity realizations (AV maps) using a variogram-driven simulation.
3. Convert each realization to depth with the depth conversion equation.
4. Compute closure, isoprobability, and volumetric distributions from stochastic depth maps.
5. Compute a mean AV map and derive one final depth map from that average AV.

This aligns with the documented workflow in:

- `referenceDocumentation/structuralUncertaintyEvaluation/Structural_Uncertainty_Workflow.txt`
- `referenceDocumentation/structuralUncertaintyEvaluation/Structural_Uncertainty_in_Petrel.pdf`

---

## 2) File Structure

Inside `structuralUncertainty_V2/`:

- `pages/structuralUncertainty_V2.py`  
  Main Panel application (UI + simulation + plotting + logging).
- `core/engine_V2.py`  
  Core simulation engine including multi-well covariance conditioning.
- `structuralUncertainty_constants_V2.py`  
  Petrel/PWR metadata, app path, and data-file constants.
- `structuralUncertainty_launcher_V2.py`  
  Petrel launcher that starts the Panel app.
- `pages/logs/`  
  Session logs generated at runtime (auto-created).

---

## 3) Runtime Flow (Step-by-Step)

### Step A — Startup and Environment

When `pages/structuralUncertainty_V2.py` is loaded:

1. Optional Windows DLL compatibility routine executes (safe no-op on Linux/macOS).
2. Repository root is injected into `sys.path` for shared theme imports.
3. OMV theme tokens and helper functions are loaded (fallback defaults exist).
4. App logger is initialized.
5. Panel extension and custom CSS are registered.

### Step B — Logging Setup

`setup_structural_logger(...)`:

1. Creates `pages/logs` if missing.
2. Creates timestamped session log file:
   - `structural_uncertainty_YYYYMMDD_HHMMSS.log`
3. Adds file handler + stdout stream handler.
4. Keeps only the newest 10 log files.
5. Writes startup log messages.

### Step C — Base Surfaces Creation

`core/engine_V2.py::build_surfaces()` returns deterministic model inputs:

- deterministic TWT map in milliseconds (ms)
- deterministic AV map in meters/second (m/s)

Supported surface modes:

- `Synthetic TWT Input` (baseline dome)
- `Elongated / Ellipsoidal` (anisotropic closure with user-controlled major/minor axis and azimuth)

### Step D — Geostatistical Realizations

The simulation pipeline:

1. Build covariance field from chosen variogram model:
   - Gaussian / Exponential / Spherical
2. Generate spatially correlated random field in frequency domain.
3. Add nugget component.
4. Optionally smooth with Gaussian filter.
5. Normalize and scale by velocity uncertainty slider (`Velocity Std. Dev. (m/s)`).

Performance details:

- stochastic field generation is handled in `core/engine_V2.py`
- covariance FFT spectrum is cached per variogram/shape setting (avoids recomputation across realizations)
- FFT operations use `scipy.fft` with multi-core workers for faster transforms
- optional Torch FFT path is used when enabled and available
- app-level stack and trap caches avoid recomputation when parameters are unchanged
- structural section extraction is vectorized across all realizations (single interpolator call)
- trap volumetrics avoids full-map temporary allocations in per-realization loop
- Monte Carlo progress is exposed in the UI progress bar

### Step E — Depth Conversion

For each stochastic velocity realization:

Depth(m) = TWT(ms) × AV(m/s) / 2000

Why divide by 2000:

- divide by 1000 to convert ms → s
- divide by 2 because TWT is two-way time, depth uses one-way travel time

So final divisor is 1000 × 2 = 2000.

### Step F — Derived Outputs

From stochastic depth stack:

- structural section (all realizations + selected realization)
- isoprobability map (percentage of realizations where the node is inside the isolated trap polygon)
- volumetrics distributions (Thickness, Area, GRV, STOIIP)

From mean AV map:

- average AV map
- final depth map derived from mean AV

---

## 4) UI Panels and Their Role

### Input Selection

- Select Surface (currently synthetic TWT input)
- Contour range
- Colormap
- Generate/Update TWT Input button

### Variogram Parameters

- Model type: Gaussian / Spherical / Exponential
- Range
- Sill
- Nugget
- Update Variogram button

Default V2 variogram starting point:

- Range = 1500 m
- Sill = 1.0
- Nugget = 0.1

### Closure Controls

- contour search step (spill-point stepping increment)
- Closure masking toggle
- Red closure contour toggle
- Update Closure/Culmination button

### Velocity Realizations

- Number of realizations
- Velocity standard deviation
- Smoothing sigma
- Section Y-range
- Realization selector
- Section/Cut angle (°)
- optional Torch acceleration toggle
- Update Velocity & Depth Realizations button
- Monte Carlo progress bar

### GRV and STOIIP

- thickness mean and thickness standard deviation
- N/G, porosity, water saturation, FVF
- Run Monte Carlo Volumetrics button

### Wells Conditioning (V2)

- Table editor for multiple wells (`Well Name`, `X`, `Y`, `Velocity`)
- `Use wells` toggle to enable stochastic conditioning
- `Show wells` toggle to display yellow well markers on maps and structural section
- Well names are drawn on maps/section and included in report figures and Excel diagnostics
- Hovering well markers in TWT / AV / Final Depth maps shows well name and XY coordinates
- Well location display in structural section is projected using the same nearest-grid anchor logic used in velocity conditioning (eliminates marker/index drift in section views and report images)
- Apply rule: after edits/toggles, press `Update Well Location` to apply

---

## 5) Plot Layout (Main Grid)

The app uses a 2×3 grid:

Top row:

1. Input Time Surface (TWT)
2. Structural Section (Depth Realizations)
3. Monte Carlo Distributions

Bottom row:

1. AV Mean and Final Depth (tabbed)
2. Isoprobability Map
3. Monte Carlo Distributions (continues full-height in right column)

Sizing notes:

- Grid and card wrappers enforce stretch behavior.
- Right-column histograms use a 4-row GridSpec to fill panel height.
- CSS is tuned so content fits in viewport with minimal scrolling.

---

## 6) Update Triggers (Reactive Behavior)

Important dependency behavior:

- Pressing Update Variogram triggers recomputation of downstream outputs:
  - structural section
  - AV/final depth tabs
  - isoprobability map
  - volumetrics

- Well table edits and well toggles also trigger end-to-end refresh so realizations, sections, maps, and histograms remain synchronized.

- Pressing Update Velocity & Depth Realizations also refreshes stochastic outputs.

- Realization slider bounds auto-follow the realization count slider.

- General settings (units) now use explicit apply behavior:
  - change unit controls,
  - then press `Apply General Settings`.

- Surface mode changes are staged and applied only after pressing `Generate / Update TWT Input`.

---

## 7) Volumetrics Logic

For each depth realization:

1. Find realization-specific culmination (crest) from the simulated depth map.
2. Step depth downward by the contour increment and isolate only the polygon connected to the crest.
3. Detect spill when that connected polygon touches map boundaries; previous level is the spill point.
4. Compute area and crest-to-spill thickness for that isolated trap only.
5. Compute total trapped volume to spill, then apply stochastic reservoir thickness to compute base volume.
6. Compute `GRV = TotalVolume - BaseVolume`.
7. Convert to HIIP/STOIIP with:

HIIP = (GRV × N/G × Φ × (1 - Sw)) / FVF

Then build histograms with:

- Mean
- P90
- P50
- P10

---

## 8) Constants and Launcher Workflow

### `structuralUncertainty_constants_V2.py`

Defines:

- workflow metadata (name, category, description, author, version)
- app path used by launcher
- temporary data-file path for runtime exchange

### `structuralUncertainty_launcher_V2.py`

Petrel/PWR integration flow:

1. Load constants.
2. Register workflow description.
3. Optionally clean old port process.
4. Write snapshot data JSON.
5. Launch Panel server:
  - serves `pages/structuralUncertainty_V2.py`
6. Opens browser to module URL.
7. Streams app logs while process is alive.

### V2 QC/Report Additions

- PDF export now includes dedicated geostatistical QC pages:
  - covariance matrix visualization + diagnostics
  - empirical vs theoretical variogram comparison
  - engineering coherence checks
  - recommended parameter table (Range / Sill / Nugget)
- Excel QC export now includes expanded sheets:
  - `Wells_Input` (well values and nearest-grid diagnostics)
  - `Wells_Covariance` and `Wells_Covariance_QC`
  - `Empirical_Variogram`
  - `Theoretical_Variogram`
  - `Geostat_Recommendations`
  - `Engineering_Coherence`

---

## 9) How to Run (Local)

From repository root:

```bash
cd /home/roderickperez/DataScienceProjects/Prizm_OMV
source .venv/bin/activate
panel serve structuralUncertainty_V2/pages/structuralUncertainty_V2.py --autoreload --show --port 5014
```

URL:

```text
http://localhost:5014/structuralUncertainty_V2
```

If `--show` cannot open browser automatically in your environment, open the URL manually.

---

## 10) Troubleshooting

### App starts but browser does not open

- This is often a local desktop/WSL browser integration issue.
- Server can still be healthy; open the URL manually.

### Plot not updating after control changes

- For variogram-sensitive changes, press Update Variogram.
- For stochastic realization updates, press Update Velocity & Depth Realizations.

### Too many logs

- Log retention is automatic; only newest 10 session files are kept in `pages/logs`.

### Import errors

Ensure environment includes:

- panel
- holoviews
- hvplot
- xarray
- numpy
- pandas
- scipy

---

## 11) Developer Notes for Future Extensions

Recommended extension points:

1. Replace synthetic TWT/velocity inputs with Petrel-selected real surfaces.
2. Add data I/O adapters for external map formats.
3. Add explicit realization export (AV and depth stacks).
4. Add QC widgets for map statistics and sanity checks.
5. Add unit tests for conversion and volumetrics helper functions.

Keep these invariants unchanged unless requirements change:

- deterministic input surface is TWT
- stochastic simulation is applied to velocity
- depth conversion uses `TWT*AV/2000`

---

## 12) Current Functionalities Summary

The current module provides the following operational capabilities:

- Input modes:
  - Synthetic baseline TWT
  - Elongated/ellipsoidal synthetic TWT
  - Imported Petrel surface (including `--test` file mode)
- Full uncertainty workflow:
  - Variogram-driven velocity realizations
  - Mean AV map and final depth map
  - Structural section extraction along configurable cut angle
  - Isoprobability map for isolated closure occurrence
  - Monte Carlo distributions for thickness, area, GRV, and STOOIP
- Trap and closure handling:
  - Domain-aware culmination/spill detection (positive and negative depth domains)
  - Closure masking and deterministic spill contour overlays
  - Final depth contour controls (toggle + contour interval)
- Unit-system controls (General Setting):
  - Velocity: ft/s or m/s
  - X/Y: meters or feet
  - Time maps: milliseconds or seconds
  - Depth maps: feet or meters
  - Area: m2 or ft2
  - Volume: m3 or ft3
  - Unit changes propagate to calculations, map labels, section labels, and reporting
- Performance and reliability:
  - Optional Torch acceleration path
  - Stack/trap caching for repeated parameter states
  - Progress bar updates for Monte Carlo processing
  - Test-mode diagnostics and logging for finite/range checks
- Reporting and export:
  - PNG/JPEG dashboard export
  - Multi-page PDF export with summary + full per-realization tables + geostatistical QC pages
  - Styled PDF tables with OMV branding
  - Companion QC Excel export generated with report (or standalone via `Export QC Excel`)
  - Petrel surface export for final depth map
