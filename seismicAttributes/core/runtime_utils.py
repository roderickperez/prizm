import ctypes
import os
import platform
import sys


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
                if not os.path.exists(dll_file):
                    continue
                try:
                    ctypes.CDLL(dll_file)
                except Exception:
                    pass
    except Exception:
        pass
