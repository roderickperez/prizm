import ctypes
import inspect
import os
import platform
import sys
import threading
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from shared.ui.omv_theme import (
    BLUE_OMV_COLOR,
    DARK_BLUE_OMV_COLOR,
    LIGHT_BLUE_OMV_COLOR,
    LIGHT_GREY_OMV_COLOR,
    LIGHT_MAGENTA_OMV_COLOR,
    NEON_OMV_COLOR,
    docs_button_html,
    get_extension_raw_css,
    get_main_outer_background,
    get_neon_button_stylesheets,
    is_dark_mode_from_state,
)


# ==============================================================================
#  WINDOWS DLL COMPATIBILITY FIX
# ==============================================================================
def apply_dll_fix() -> None:
    if platform.system() != "Windows":
        return

    try:
        current_venv = sys.prefix
        torch_lib_path = os.path.join(current_venv, "Lib", "site-packages", "torch", "lib")
        user_lib_path = os.path.expanduser(r"~\py_pkgs\torch\lib")

        for lib_path in [torch_lib_path, user_lib_path]:
            if not os.path.exists(lib_path):
                continue

            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(lib_path)
                except Exception:
                    pass

            for dll in ["libiomp5md.dll", "c10.dll", "torch_python.dll"]:
                dll_file = os.path.join(lib_path, dll)
                if os.path.exists(dll_file):
                    try:
                        ctypes.CDLL(dll_file)
                    except Exception:
                        pass
    except Exception:
        pass


apply_dll_fix()


# ==============================================================================
#  IMPORTS AFTER DLL FIX
# ==============================================================================
import panel as pn
import torch


# ==============================================================================
#  APP CONFIG
# ==============================================================================
APP_TITLE = "GPU Diagnostic Tool"
DOCUMENTATION_URL = "https://example.com/docs"
SCREEN_CODE_COLOR = "#263238"

MATRIX_SIZE = max(512, int(os.environ.get("GPU_BENCHMARK_MATRIX_SIZE", "2048")))
BENCHMARK_STEPS = max(1, int(os.environ.get("GPU_BENCHMARK_STEPS", "3")))

ASSETS_DIR = ROOT_DIR / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

is_dark_mode = is_dark_mode_from_state()


# ==============================================================================
#  PANEL EXTENSION / CSS
# ==============================================================================
gpu_css = f"""
.scrollable-code {{
    overflow-y: auto !important;
    height: 100% !important;
    max-height: none !important;
    border: 1px solid #ddd !important;
    padding: 10px !important;
    background-color: {SCREEN_CODE_COLOR} !important;
}}
.scrollable-code pre,
.scrollable-code code {{
    background-color: {SCREEN_CODE_COLOR} !important;
}}
.black-text-log textarea {{
    color: black !important;
    font-family: monospace !important;
}}
.cpu-bg {{
    background-color: {LIGHT_MAGENTA_OMV_COLOR} !important;
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 8px;
}}
.gpu-bg {{
    background-color: {LIGHT_BLUE_OMV_COLOR} !important;
    border: 1px solid #90caf9;
    padding: 15px;
    border-radius: 8px;
}}
.app-main-fit {{
    display: grid !important;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr) !important;
    gap: 12px !important;
    height: calc(100vh - 120px) !important;
    overflow: hidden !important;
}}
.bench-bottom {{
    margin-top: 0 !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
    align-items: stretch !important;
}}
.code-fill {{
    min-height: 0 !important;
    overflow: hidden !important;
}}
.code-fill .bk-Card,
.code-fill .bk-panel-models-layout-Card,
.code-fill .bk-card-body,
.code-fill .bk-Column,
.code-fill .bk-card,
.code-fill .scrollable-code {{
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
}}
.code-fill .scrollable-code {{
    overflow-y: auto !important;
}}
.runner-log textarea {{
    min-height: 0 !important;
    max-height: none !important;
    height: 100% !important;
}}
.cpu-bg,
.gpu-bg {{
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 0 !important;
}}
.bench-bottom .bk-Column,
.bench-bottom .bk-panel-models-layout-Column {{
    height: 100% !important;
    min-height: 0 !important;
}}
.runner-log {{
    flex: 1 1 auto !important;
    min-height: 0 !important;
}}
.omv-run-btn,
.omv-run-btn.bk-btn,
button.omv-run-btn,
.omv-run-btn .bk-btn,
.omv-run-btn button.bk-btn {{
    background: {NEON_OMV_COLOR} !important;
    color: {DARK_BLUE_OMV_COLOR} !important;
    border-color: {NEON_OMV_COLOR} !important;
    font-weight: 600 !important;
}}

.omv-run-btn .bk-btn:hover,
.omv-run-btn:hover,
.omv-run-btn.bk-btn:hover,
button.omv-run-btn:hover,
.omv-run-btn button.bk-btn:hover {{
    filter: brightness(0.96);
}}
.dark-mode-main-text,
.dark-mode-main-text .bk,
.dark-mode-main-text .bk-root,
.dark-mode-main-text .bk-Card-title,
.dark-mode-main-text .bk-panel-models-markup-Markdown,
.dark-mode-main-text .bk-panel-models-markup-HTML {{
    color: {DARK_BLUE_OMV_COLOR} !important;
}}

.dark-mode-main-text .runner-log textarea {{
    color: black !important;
}}

.dark-mode-main-text .omv-run-btn,
.dark-mode-main-text .omv-run-btn.bk-btn,
.dark-mode-main-text button.omv-run-btn,
.dark-mode-main-text .omv-run-btn .bk-btn,
.dark-mode-main-text .omv-run-btn button.bk-btn {{
    background: {NEON_OMV_COLOR} !important;
    color: {DARK_BLUE_OMV_COLOR} !important;
    border-color: {NEON_OMV_COLOR} !important;
    font-weight: 600 !important;
}}

.dark-mode-main-text .omv-run-btn:hover,
.dark-mode-main-text .omv-run-btn.bk-btn:hover,
.dark-mode-main-text button.omv-run-btn:hover,
.dark-mode-main-text .omv-run-btn button.bk-btn:hover {{
    filter: brightness(0.96);
}}
"""

pn.extension("tabulator", raw_css=get_extension_raw_css(is_dark_mode) + [gpu_css])


# ==============================================================================
#  GPU DIAGNOSTICS
# ==============================================================================
def get_gpu_details() -> dict[str, str | bool]:
    info: dict[str, str | bool] = {
        "Available": False,
        "Name": "N/A",
        "VRAM": "0 GB",
        "Cuda Ver": "N/A",
        "Capability": "N/A",
    }

    try:
        if not torch.cuda.is_available():
            return info

        device_index = 0
        props = torch.cuda.get_device_properties(device_index)
        total_mem_gb = round(props.total_memory / (1024 ** 3), 2)

        info["Available"] = True
        info["Name"] = torch.cuda.get_device_name(device_index)
        info["VRAM"] = f"{total_mem_gb} GB"
        info["Cuda Ver"] = torch.version.cuda or "N/A"
        info["Capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        print(f"Error reading GPU properties: {exc}")

    return info


gpu_stats = get_gpu_details()


# ==============================================================================
#  BENCHMARK LOGIC
# ==============================================================================
def _sync_if_cuda(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


def run_benchmark_logic(device_type: str, progress_callback=None, log_callback=None) -> None:
    if log_callback:
        log_callback(f"--- Starting {device_type.upper()} Benchmark ---")
        log_callback(f"Task: Matrix Mul ({MATRIX_SIZE}x{MATRIX_SIZE})")

    if device_type == "cuda" and not torch.cuda.is_available():
        if log_callback:
            log_callback("ERROR: CUDA not available!")
        return

    try:
        device = torch.device(device_type)

        if log_callback:
            log_callback("1. Allocating Data (Setup)...")

        setup_start = time.perf_counter()
        with torch.inference_mode():
            x = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
            y = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        _sync_if_cuda(device_type)
        setup_time = time.perf_counter() - setup_start

        if log_callback:
            log_callback(f"   -> Setup Time: {setup_time:.4f} sec")

        if device_type == "cuda":
            if log_callback:
                log_callback("2. Warming up GPU...")
            with torch.inference_mode():
                _ = torch.matmul(x, y)
            _sync_if_cuda(device_type)

        if log_callback:
            log_callback(f"3. Running {BENCHMARK_STEPS} Compute Steps...")

        compute_start = time.perf_counter()
        with torch.inference_mode():
            for step_idx in range(BENCHMARK_STEPS):
                step_start = time.perf_counter()
                _ = torch.matmul(x, y)
                _sync_if_cuda(device_type)
                step_elapsed = time.perf_counter() - step_start

                if log_callback:
                    log_callback(f"   Step {step_idx + 1}: {step_elapsed:.4f} sec")
                if progress_callback:
                    progress_callback(int((step_idx + 1) / BENCHMARK_STEPS * 100))

        compute_time = time.perf_counter() - compute_start
        avg_time = compute_time / BENCHMARK_STEPS

        if log_callback:
            log_callback("-" * 30)
            log_callback(f"SETUP TIME:   {setup_time:.4f} sec")
            log_callback(f"COMPUTE TIME: {compute_time:.4f} sec")
            log_callback(f"AVG PER STEP: {avg_time:.4f} sec")
            log_callback("-" * 30)
    except Exception as exc:
        if log_callback:
            log_callback(f"CRITICAL ERROR: {exc}")


# ==============================================================================
#  UI BUILDERS
# ==============================================================================
def create_runner_column(title: str, device_type: str) -> pn.Column:
    button_type = "default"
    run_button = pn.widgets.Button(
        name=f"Run {title}",
        button_type=button_type,
        icon="player-play",
        css_classes=["omv-run-btn"],
        stylesheets=get_neon_button_stylesheets(),
    )
    progress_bar = pn.indicators.Progress(name="Progress", value=0, width=220, visible=False)
    console_output = pn.widgets.TextAreaInput(
        value="Ready...",
        disabled=False,
        sizing_mode="stretch_width",
        css_classes=["black-text-log", "runner-log"],
    )

    logs: list[str] = ["Ready..."]

    def append_log(message: str) -> None:
        logs.append(message)
        console_output.value = "\n".join(logs[-400:])

    def execute_test(_event) -> None:
        doc = pn.state.curdoc

        def schedule_ui_update(func) -> None:
            if doc is not None:
                doc.add_next_tick_callback(func)

        run_button.disabled = True
        progress_bar.visible = True
        progress_bar.value = 0
        logs.clear()
        append_log(f"Initializing {title}...")

        def update_progress(value: int) -> None:
            schedule_ui_update(lambda: setattr(progress_bar, "value", value))

        def thread_log(message: str) -> None:
            schedule_ui_update(lambda: append_log(message))

        def finalize_ui() -> None:
            run_button.disabled = False
            progress_bar.visible = False

        def run_in_thread() -> None:
            try:
                run_benchmark_logic(device_type, progress_callback=update_progress, log_callback=thread_log)
            except Exception as exc:
                thread_log(f"Error: {exc}")
            finally:
                schedule_ui_update(finalize_ui)

        threading.Thread(target=run_in_thread, daemon=True).start()

    run_button.on_click(execute_test)

    container_class = "cpu-bg" if device_type == "cpu" else "gpu-bg"
    title_color = DARK_BLUE_OMV_COLOR if is_dark_mode else BLUE_OMV_COLOR
    return pn.Column(
        pn.pane.HTML(f"<h3 style='margin:0 0 8px 0; color:{title_color};'>{title}</h3>"),
        run_button,
        progress_bar,
        console_output,
        css_classes=[container_class],
        sizing_mode="stretch_both",
    )


# ==============================================================================
#  MAIN CONTENT
# ==============================================================================
code_source = inspect.getsource(run_benchmark_logic)
code_pane = pn.Column(
    pn.pane.Markdown(f"```python\n{code_source}\n```", sizing_mode="stretch_width"),
    css_classes=["scrollable-code"],
    sizing_mode="stretch_both",
)

code_card = pn.Card(
    code_pane,
    title="Benchmark Logic (Python code)",
    collapsed=True,
    hide_header=False,
    sizing_mode="stretch_both",
    header_background=NEON_OMV_COLOR if is_dark_mode else LIGHT_GREY_OMV_COLOR,
    active_header_background=NEON_OMV_COLOR if is_dark_mode else LIGHT_GREY_OMV_COLOR,
    header_color=DARK_BLUE_OMV_COLOR,
    css_classes=["code-fill"],
)

cpu_col = create_runner_column("CPU (Baseline)", "cpu")
gpu_col = create_runner_column("GPU (Nvidia)", "cuda")

benchmark_row = pn.Row(
    cpu_col,
    pn.Spacer(width=20),
    gpu_col,
    sizing_mode="stretch_width",
    css_classes=["bench-bottom"],
)

main_content = pn.Column(
    code_card,
    benchmark_row,
    sizing_mode="stretch_both",
    css_classes=["app-main-fit", "dark-mode-main-text"] if is_dark_mode else ["app-main-fit"],
    margin=0,
    styles={
        "height": "100%",
        "overflow": "hidden",
        "background": get_main_outer_background(is_dark_mode),
        "color": DARK_BLUE_OMV_COLOR if is_dark_mode else "inherit",
    },
)

status_color = "green" if gpu_stats["Available"] else "red"
status_text = "[ACTIVE]" if gpu_stats["Available"] else "[INACTIVE]"

sidebar_items = [
    pn.pane.Markdown("### GPU Dashboard"),
    pn.pane.Markdown("**Status:**"),
    pn.pane.HTML(f"<h3 style='color:{status_color}; margin:0;'>{status_text}</h3>"),
    pn.pane.Markdown(f"**Model:** {gpu_stats['Name']}"),
    pn.pane.Markdown(f"**VRAM:** {gpu_stats['VRAM']}"),
    pn.pane.Markdown(f"**CUDA:** {gpu_stats['Cuda Ver']}"),
    pn.pane.Markdown(f"**Capability:** {gpu_stats['Capability']}"),
    pn.pane.Markdown("---"),
    pn.pane.Markdown("### Env Info"),
    pn.pane.Markdown(f"**Python:** {sys.version.split(' ')[0]}"),
    pn.pane.Markdown(f"**Torch:** {torch.__version__}"),
    pn.pane.Markdown(f"**Matrix Size:** {MATRIX_SIZE}"),
    pn.pane.Markdown(f"**Steps:** {BENCHMARK_STEPS}"),
]

valid_logo = str(LOGO_PATH) if LOGO_PATH.exists() else None
valid_favicon = str(FAVICON_PATH) if FAVICON_PATH.exists() else None

template_kwargs = dict(
    title=APP_TITLE,
    accent_base_color=BLUE_OMV_COLOR,
    header_background=DARK_BLUE_OMV_COLOR,
    main_layout=None,
    main_max_width="",
    sidebar=sidebar_items,
    main=[main_content],
    header=[
        pn.Row(
            pn.Spacer(sizing_mode="stretch_width"),
            pn.pane.HTML(docs_button_html(DOCUMENTATION_URL)),
            sizing_mode="stretch_width",
            margin=0,
        )
    ],
)

if valid_logo:
    template_kwargs["logo"] = valid_logo
if valid_favicon:
    template_kwargs["favicon"] = valid_favicon

pn.template.FastListTemplate(**template_kwargs).servable()
