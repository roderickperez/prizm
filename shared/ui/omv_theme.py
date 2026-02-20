import panel as pn

NEON_OMV_COLOR = "#21ff59"
GREEN_OMV_COLOR = "#007733"
DARK_GREEN_OMV_COLOR = "#195b42"
LIGHT_GREEN_OMV_COLOR = "#99ffb3"

LIGHT_BLUE_OMV_COLOR = "#acf7fb"
BLUE_OMV_COLOR = "#335290"
DARK_BLUE_OMV_COLOR = "#042758"
NEON_BLUE_OMV_COLOR = "#02ecfe"

LIGHT_MAGENTA_OMV_COLOR = "#ecdaff"
MAGENTA_OMV_COLOR = "#8848ea"
DARK_MAGENTA_OMV_COLOR = "#4536a2"
NEON_MAGENTA_OMV_COLOR = "#d8b7fc"

LIGHT_GREY_OMV_COLOR = "#f2f2f2"

DARK_MODE_MAIN_BACKGROUND = DARK_GREEN_OMV_COLOR
DARK_MODE_CONTENT_BACKGROUND = DARK_BLUE_OMV_COLOR
DARK_MODE_OUTER_BACKGROUND = LIGHT_BLUE_OMV_COLOR
DARK_MODE_PLOT_BACKGROUND = DARK_BLUE_OMV_COLOR


def is_dark_mode_from_state() -> bool:
    theme_arg = pn.state.session_args.get("theme", [b"default"])
    theme_value = theme_arg[0].decode("utf-8") if theme_arg else "default"
    return theme_value == "dark"


def get_dark_select_stylesheets(is_dark_mode: bool) -> list[str]:
    if not is_dark_mode:
        return []
    return [
        f"""
        :host {{
            --input-background: {GREEN_OMV_COLOR};
            --input-color: white;
            --input-border-color: {GREEN_OMV_COLOR};
        }}

        select,
        .bk-input,
        .bk-input-group select,
        .bk-input-group .bk-input {{
            background: {GREEN_OMV_COLOR} !important;
            color: white !important;
            border-color: {GREEN_OMV_COLOR} !important;
        }}

        option {{
            background: {GREEN_OMV_COLOR} !important;
            color: white !important;
        }}
        """
    ]


def get_slider_stylesheets() -> list[str]:
    return [
        f"""
        :host {{
            --handle-color: {DARK_GREEN_OMV_COLOR};
            --slider-handle-color: {DARK_GREEN_OMV_COLOR};
        }}

        .noUi-handle,
        .noUi-handle:before,
        .noUi-handle:after,
        .noUi-touch-area,
        .bk-slider-handle,
        .bk-noUi-handle,
        [class*='noUi-handle'] {{
            background: {DARK_GREEN_OMV_COLOR} !important;
            border-color: {DARK_GREEN_OMV_COLOR} !important;
            box-shadow: none !important;
        }}
        """
    ]


def get_neon_button_stylesheets() -> list[str]:
    return [
        f"""
        :host {{
            --button-color: {DARK_BLUE_OMV_COLOR};
            --button-border-color: {NEON_OMV_COLOR};
            --button-bg: {NEON_OMV_COLOR};
            --button-background: {NEON_OMV_COLOR};
        }}

        .bk-btn,
        .bk-btn.bk-btn-default,
        .bk-btn.bk-btn-primary,
        .bk-btn.bk-btn-success,
        .bk-btn.bk-btn-warning,
        button.bk-btn {{
            background: {NEON_OMV_COLOR} !important;
            background-color: {NEON_OMV_COLOR} !important;
            color: {DARK_BLUE_OMV_COLOR} !important;
            border-color: {NEON_OMV_COLOR} !important;
            font-weight: 600 !important;
        }}

        .bk-btn:hover,
        .bk-btn.bk-btn-default:hover,
        .bk-btn.bk-btn-primary:hover,
        .bk-btn.bk-btn-success:hover,
        .bk-btn.bk-btn-warning:hover,
        button.bk-btn:hover {{
            filter: brightness(0.96);
        }}
        """
    ]


def get_radio_button_stylesheets() -> list[str]:
    return [
        f"""
        .bk-btn-group .bk-btn,
        .bk-btn-group button.bk-btn,
        .bk-btn-group .bk-btn:not(.bk-active),
        .bk-btn-group button.bk-btn:not(.bk-active),
        .bk-btn-group .bk-btn[aria-pressed='false'],
        .bk-btn-group button.bk-btn[aria-pressed='false'] {{
            background: {NEON_MAGENTA_OMV_COLOR} !important;
            background-color: {NEON_MAGENTA_OMV_COLOR} !important;
            border-color: {MAGENTA_OMV_COLOR} !important;
            color: {DARK_BLUE_OMV_COLOR} !important;
            font-weight: 400 !important;
        }}

        .bk-btn-group .bk-btn.bk-active,
        .bk-btn-group button.bk-btn.bk-active,
        .bk-btn-group .bk-btn[aria-pressed='true'],
        .bk-btn-group button.bk-btn[aria-pressed='true'] {{
            background: {MAGENTA_OMV_COLOR} !important;
            background-color: {MAGENTA_OMV_COLOR} !important;
            border-color: {MAGENTA_OMV_COLOR} !important;
            color: white !important;
            font-weight: 400 !important;
        }}
        """
    ]


def get_plot_surface_background(is_dark_mode: bool) -> str:
    return DARK_MODE_PLOT_BACKGROUND if is_dark_mode else "white"


def get_main_outer_background(is_dark_mode: bool) -> str:
    return DARK_MODE_OUTER_BACKGROUND if is_dark_mode else LIGHT_GREY_OMV_COLOR


def get_content_text_color(is_dark_mode: bool) -> str:
    return "white" if is_dark_mode else DARK_BLUE_OMV_COLOR


def get_dark_colorbar_opts(is_dark_mode: bool) -> dict:
    if not is_dark_mode:
        return {}
    return {
        "background_fill_color": DARK_MODE_PLOT_BACKGROUND,
        "background_fill_alpha": 1.0,
        "border_line_color": DARK_MODE_PLOT_BACKGROUND,
        "major_label_text_color": "white",
        "title_text_color": "white",
    }


def get_section_card_colors(is_dark_mode: bool) -> dict[str, str]:
    return {
        "header_background": NEON_OMV_COLOR,
        "header_text": DARK_BLUE_OMV_COLOR,
        "body_background": DARK_BLUE_OMV_COLOR if is_dark_mode else "white",
        "body_text": "white" if is_dark_mode else "inherit",
    }


def get_dark_text_input_stylesheets(is_dark_mode: bool) -> list[str]:
    if not is_dark_mode:
        return []
    return [
        f"""
        :host {{
            --input-background: {DARK_GREEN_OMV_COLOR};
            --input-color: white;
            --input-border-color: {DARK_GREEN_OMV_COLOR};
        }}

        input,
        input[type="number"],
        textarea,
        .bk-input,
        .bk-input[type="number"],
        .bk-Spinner input,
        .bk-Spinner .bk-input,
        .bk-input-group .bk-Spinner input,
        .bk-input-group input,
        .bk-input-group .bk-input {{
            background: {DARK_GREEN_OMV_COLOR} !important;
            background-color: {DARK_GREEN_OMV_COLOR} !important;
            color: white !important;
            border-color: {DARK_GREEN_OMV_COLOR} !important;
        }}

        .bk-Spinner button,
        .bk-Spinner .bk-btn,
        .bk-input-group .bk-Spinner button,
        .bk-input-group .bk-Spinner .bk-btn {{
            background: {DARK_GREEN_OMV_COLOR} !important;
            background-color: {DARK_GREEN_OMV_COLOR} !important;
            color: white !important;
            border-color: {DARK_GREEN_OMV_COLOR} !important;
        }}

        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button {{
            background: {DARK_GREEN_OMV_COLOR} !important;
            color: white !important;
        }}
        """
    ]


def _docs_button_css() -> str:
    return (
        """
        .docs-button .bk-btn,
        .docs-button .bk-btn.bk-btn-default,
        .docs-button .bk-btn.bk-btn-primary,
        .docs-button button.bk-btn {
            background-color: __NEON_OMV_COLOR__ !important;
            color: __BLUE_OMV_COLOR__ !important;
            border-color: __NEON_OMV_COLOR__ !important;
            font-weight: 600;
        }

        .docs-button .bk-btn:hover,
        .docs-button .bk-btn.bk-btn-default:hover,
        .docs-button .bk-btn.bk-btn-primary:hover,
        .docs-button button.bk-btn:hover {
            filter: brightness(0.96);
        }
        """
        .replace("__NEON_OMV_COLOR__", NEON_OMV_COLOR)
        .replace("__BLUE_OMV_COLOR__", BLUE_OMV_COLOR)
    )


def _dark_mode_css() -> str:
    return (
        """
#container,
#content,
#main,
#main .pn-wrapper,
#main .card-margin,
#main .main-margin {
    background: __DARK_MODE_MAIN_BACKGROUND__ !important;
}

#sidebar,
#sidebar .nav,
#sidebar .pn-wrapper,
#sidebar .bk-Column {
    background: __MAGENTA_OMV_COLOR__ !important;
}

#sidebar .bk-Card,
#sidebar .bk-panel-models-layout-Card,
#sidebar .bk-panel-models-widgets-box-Box {
    background: __DARK_BLUE_OMV_COLOR__ !important;
    color: white !important;
}

#sidebar .bk-input,
#sidebar input[type='number'],
#sidebar select,
#sidebar .bk-input-group input,
#sidebar .bk-input-group select {
    background: __BLUE_OMV_COLOR__ !important;
    color: white !important;
    border-color: __LIGHT_BLUE_OMV_COLOR__ !important;
}

#sidebar .bk-Select .bk-input,
#sidebar select,
#sidebar .bk-input,
#sidebar input[type='number'],
#sidebar .bk-Spinner input,
#sidebar .bk-input-group select {
    background: __DARK_GREEN_OMV_COLOR__ !important;
    color: white !important;
    border-color: __DARK_GREEN_OMV_COLOR__ !important;
    -webkit-appearance: none;
    appearance: none;
}

#sidebar .bk-Spinner button,
#sidebar .bk-Spinner .bk-btn {
    background: __DARK_GREEN_OMV_COLOR__ !important;
    color: white !important;
    border-color: __DARK_GREEN_OMV_COLOR__ !important;
}

#sidebar .bk-input option,
#sidebar select option {
    background: __DARK_GREEN_OMV_COLOR__ !important;
    color: white !important;
}

#sidebar .bk-slider-title,
#sidebar .bk-form-group label,
#sidebar .bk-root,
#sidebar .bk {
    color: white !important;
}

#sidebar .noUi-target,
#sidebar .noUi-base,
#sidebar .noUi-connects {
    background: __LIGHT_BLUE_OMV_COLOR__ !important;
}

#sidebar .noUi-connect {
    background: __NEON_OMV_COLOR__ !important;
}
        """
        .replace("__DARK_MODE_MAIN_BACKGROUND__", DARK_MODE_MAIN_BACKGROUND)
        .replace("__BLUE_OMV_COLOR__", BLUE_OMV_COLOR)
        .replace("__LIGHT_BLUE_OMV_COLOR__", LIGHT_BLUE_OMV_COLOR)
        .replace("__MAGENTA_OMV_COLOR__", MAGENTA_OMV_COLOR)
        .replace("__DARK_GREEN_OMV_COLOR__", DARK_GREEN_OMV_COLOR)
        .replace("__DARK_BLUE_OMV_COLOR__", DARK_BLUE_OMV_COLOR)
        .replace("__NEON_OMV_COLOR__", NEON_OMV_COLOR)
    )


def _light_mode_css() -> str:
    return f"""
#sidebar .noUi-handle,
#sidebar .noUi-handle:before,
#sidebar .noUi-handle:after,
#sidebar .noUi-touch-area {{
    background: {DARK_GREEN_OMV_COLOR} !important;
    border-color: {DARK_GREEN_OMV_COLOR} !important;
}}
"""


def get_extension_raw_css(is_dark_mode: bool) -> list[str]:
    return [_docs_button_css(), _dark_mode_css() if is_dark_mode else _light_mode_css()]


def docs_button_html(url: str) -> str:
    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        f'style="display:inline-block; padding:6px 14px; border-radius:4px; text-decoration:none; '
        f'background:{NEON_OMV_COLOR}; color:{BLUE_OMV_COLOR}; font-weight:600;">Documentation</a>'
    )
