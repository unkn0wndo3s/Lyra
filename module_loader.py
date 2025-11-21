"""
Dynamic loader for Python modules inside the local ``modules`` directory.

Features:
    * Automatically loads every module under ``modules/`` when the script starts.
    * Ability to reload every previously loaded module without stopping the script.
    * Ability to reload a single module by name.
    * Ability to load only the modules that have been added since the script started.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable

BASE_DIR = Path(__file__).parent
MODULES_PACKAGE = "modules"
MODULES_PATH = BASE_DIR / MODULES_PACKAGE


@dataclass
class LoadedModule:
    """Container that keeps the module and its exported symbols."""

    module: ModuleType
    classes: Dict[str, type]
    functions: Dict[str, Callable]


LOADED_MODULES: Dict[str, LoadedModule] = {}


def _ensure_modules_path_on_sys_path() -> None:
    """Ensure Python can resolve the local modules package."""
    parent = str(BASE_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _extract_symbols(module: ModuleType) -> tuple[Dict[str, type], Dict[str, Callable]]:
    """Collect classes and functions defined inside a module."""
    classes: Dict[str, type] = {}
    functions: Dict[str, Callable] = {}

    for name, obj in vars(module).items():
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            classes[name] = obj
        elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
            functions[name] = obj

    return classes, functions


def _import_module(module_name: str) -> LoadedModule:
    """Import a module from the modules package and track its symbols."""
    if not MODULES_PATH.exists():
        raise FileNotFoundError(f"Modules directory not found: {MODULES_PATH}")

    _ensure_modules_path_on_sys_path()
    qualified_name = f"{MODULES_PACKAGE}.{module_name}"
    module = importlib.import_module(qualified_name)
    classes, functions = _extract_symbols(module)
    loaded = LoadedModule(module=module, classes=classes, functions=functions)
    LOADED_MODULES[module_name] = loaded
    return loaded


def discover_module_names() -> list[str]:
    """Return a sorted list of module names inside the modules directory."""
    if not MODULES_PATH.exists():
        return []
    iterator = pkgutil.iter_modules([str(MODULES_PATH)])
    return sorted(module.name for module in iterator)


def load_all_modules() -> Dict[str, LoadedModule]:
    """Load every module in the modules directory."""
    for module_name in discover_module_names():
        load_module(module_name)
    return dict(LOADED_MODULES)


def load_module(module_name: str) -> LoadedModule:
    """Load a single module by name, even if it was already loaded."""
    return _import_module(module_name)


def reload_module(module_name: str) -> LoadedModule:
    """Reload a previously loaded module."""
    if module_name not in LOADED_MODULES:
        raise KeyError(f"Module '{module_name}' has not been loaded yet.")
    module = importlib.reload(LOADED_MODULES[module_name].module)
    classes, functions = _extract_symbols(module)
    updated = LoadedModule(module=module, classes=classes, functions=functions)
    LOADED_MODULES[module_name] = updated
    return updated


def reload_all_modules() -> Dict[str, LoadedModule]:
    """Reload every module that has already been loaded."""
    for module_name in list(LOADED_MODULES.keys()):
        reload_module(module_name)
    return dict(LOADED_MODULES)


def load_unloaded_modules() -> Dict[str, LoadedModule]:
    """Load only the modules that are not already loaded."""
    loaded_now: Dict[str, LoadedModule] = {}
    for module_name in discover_module_names():
        if module_name not in LOADED_MODULES:
            loaded_now[module_name] = load_module(module_name)
    return loaded_now


def summarize_loaded_modules(modules: Iterable[tuple[str, LoadedModule]]) -> str:
    """Return a human-readable summary for loaded modules."""
    lines = []
    for name, loaded in modules:
        classes = ", ".join(loaded.classes) or "No classes"
        functions = ", ".join(loaded.functions) or "No functions"
        lines.append(f"- {name}: classes [{classes}] | functions [{functions}]")
    return "\n".join(lines)


def _default_load() -> None:
    """Default behavior: load every module when the script starts."""
    load_all_modules()


def main() -> None:
    """Simple demonstration when running this file directly."""
    print("Default load:")
    print(summarize_loaded_modules(LOADED_MODULES.items()) or "No modules found.")

    print("\nReloading all modules...")
    reload_all_modules()
    print(summarize_loaded_modules(LOADED_MODULES.items()) or "No modules reloaded.")

    newly_loaded = load_unloaded_modules()
    if newly_loaded:
        print("\nModules that were loaded later:")
        print(summarize_loaded_modules(newly_loaded.items()))


# Perform the default loading as soon as the module is imported.
_default_load()


if __name__ == "__main__":
    main()

