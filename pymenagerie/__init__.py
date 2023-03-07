from pathlib import Path

__version__ = "0.0.1"

# Path to the root of the project.
_PROJECT_ROOT = Path(__file__).parent.parent

# Path to the Menagerie submodule.
MENAGERIE_ROOT = _PROJECT_ROOT / "third_party" / "mujoco_menagerie"


__all__ = [
    "__version__",
    "MENAGERIE_ROOT",
]
