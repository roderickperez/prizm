# N:\_USER_GLOBAL\PETREL\Prizm\wf\EDA\app.py

from pathlib import Path

import streamlit as st

from utils import hide_native_sidebar_pages, build_streamlit_navigation



st.set_page_config(
    page_title="GeoPython | Well EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Hide Streamlit's default pages list (belt & suspenders; also done inside utils)

hide_native_sidebar_pages()



# Register all pages from a single source of truth

nav = build_streamlit_navigation(position="sidebar", expanded=False)



# Run selected page

nav.run()