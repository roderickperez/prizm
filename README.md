# Prizm_OMV — Run Guide (uv + Panel)

This README explains exactly how to run each Panel app from terminal.

## 1) Open terminal in project root

```bash
cd /home/roderickperez/DataScienceProjects/Prizm_OMV
```

---

## 2) Create and activate the uv Python environment

### First time setup

```bash
# create virtual environment in .venv
uv venv .venv

# activate it
source .venv/bin/activate

# install dependencies
uv pip install -r requirements.txt
```

### Next times

```bash
cd /home/roderickperez/DataScienceProjects/Prizm_OMV
source .venv/bin/activate
```

Check environment:

```bash
which python
python --version
```

---

## 3) Run any app

General pattern:

```bash
panel serve <path_to_main.py> --autoreload --show --port <PORT>
```

- `--autoreload`: reloads when code changes
- `--show`: opens browser automatically
- If browser does not open, copy the URL shown in this README

> Run only **one app at a time** per port. Stop with `Ctrl+C` before starting another.

---

## 4) App-by-app commands

### A) Base App

```bash
panel serve BaseApp/panel/PanelBase_main.py --autoreload --show --port 5006
```

URL:

```text
http://localhost:5006/PanelBase_main
```

---

### B) GPU Test App

```bash
panel serve GPU_Test/panel/PanelBase_GPU_main.py --autoreload --show --port 5007
```

URL:

```text
http://localhost:5007/PanelBase_GPU_main
```

---

### C) Fault Segmentation

```bash
panel serve faultSeg/pages/faultSeg_main.py --autoreload --show --port 5006
```

URL:

```text
http://localhost:5006/faultSeg_main
```

---

### D) Seismic RGT 3D

```bash
panel serve RGTNet_3D/pages/seismicRGT_main.py --autoreload --show --port 5006
```

URL:

```text
http://localhost:5006/seismicRGT_main
```

---

### E) Seismic Attributes

```bash
panel serve seismicAttributes/pages/seismicAttributes_main.py --autoreload --show --port 5006
```

URL:

```text
http://localhost:5006/seismicAttributes_main
```

---

### F) Seismic Denoising

```bash
panel serve seismicDenoising/pages/seismicDenoising_main.py --autoreload --show --port 5006
```

URL:

```text
http://localhost:5006/seismicDenoising_main
```

---

### G) Seismic Flowlines 2D

```bash
panel serve seismicFlowline_2D/pages/seismicFlowLines_2D_main.py --autoreload --show --port 5006
```

URL:

```text
http://localhost:5006/seismicFlowLines_2D_main
```

---

### H) Seismic Footprint Suppression

```bash
panel serve seismicFootprint/pages/seismic_footprintSuppression_main.py --autoreload --show --port 5006
```

URL:

```text
http://localhost:5006/seismic_footprintSuppression_main
```

---

### I) Structural Uncertainty

```bash
panel serve structuralUncertainty/pages/structuralUncertainty.py --autoreload --show --port 5006
```

URL:

```text
http://localhost:5006/structuralUncertainty
```

---

### J) CheckShot QC

Regular mode (Petrel/PWR data or default runtime):

```bash
panel serve checkShotQC/pages/checkShot_main.py --autoreload --show --port 5014
```

URL:

```text
http://localhost:5014/checkShot_main
```

Test mode (load all split wells from `referenceDocumentation/checkShotQC/wells`):

```bash
panel serve checkShotQC/pages/checkShot_main.py --autoreload --show --port 5014 --args --test
```

Alternative test mode using env var:

```bash
PRIZM_CHECKSHOT_TEST_MODE=1 panel serve checkShotQC/pages/checkShot_main.py --autoreload --show --port 5014
```

In test mode, the app reads:

```text
referenceDocumentation/checkShotQC/wells/*.csv
```

---

## 5) Optional: pass data file from terminal (if app expects `PWR_DATA_FILE`)

Some apps read data from `PWR_DATA_FILE`.

Example:

```bash
export PWR_DATA_FILE=/absolute/path/to/data.json
panel serve BaseApp/panel/PanelBase_main.py --autoreload --show --port 5006
```

Unset after test:

```bash
unset PWR_DATA_FILE
```

---

## 6) Troubleshooting

### `panel: command not found`

```bash
source .venv/bin/activate
uv pip install panel
```

### Port already in use

Use a different port:

```bash
panel serve BaseApp/panel/PanelBase_main.py --autoreload --show --port 5010
```

URL will be:

```text
http://localhost:5010/PanelBase_main
```

### App starts but blank / errors

Run without `--show` and read terminal logs carefully:

```bash
panel serve BaseApp/panel/PanelBase_main.py --autoreload --port 5006
```

---

## 7) MCP workflow (Copilot + HoloViz MCP)

If `holoviz-mcp` is configured in VS Code:

1. Open Command Palette: `MCP: List Servers`
2. Start server: `holoviz`
3. In Copilot Chat, prefix prompts with `#holoviz`

Examples:

- `#holoviz List available Panel widgets for form input`
- `#holoviz Improve layout and responsiveness in BaseApp/panel/PanelBase_main.py`
- `#holoviz Refactor this file to cleaner Panel component structure while keeping behavior`

This lets Copilot use the HoloViz docs/tools to help you modify and improve your Panel apps.
