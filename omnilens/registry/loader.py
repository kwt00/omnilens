from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml


ARCHITECTURES_DIR = Path(__file__).parent / "architectures"


class Registry:
    """Maps omnilens standardized names to native model module paths.

    A registry is a flat dict where:
      - Keys are omnilens names (e.g. 'layers.0.attention.q')
      - Values are native module paths (e.g. 'model.layers.0.self_attn.q_proj')

    Registry entries can use '{i}' as a placeholder for layer indices.
    These get expanded when the registry is applied to a model with a known
    number of layers.
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def __getitem__(self, key: str) -> str:
        return self._mapping[key]

    def __contains__(self, key: str) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def keys(self):
        return self._mapping.keys()

    def values(self):
        return self._mapping.values()

    def items(self):
        return self._mapping.items()

    def __repr__(self) -> str:
        return f"Registry({len(self._mapping)} entries)"

    def expand_layers(self, n_layers: int) -> Registry:
        """Expand '{i}' placeholders into concrete layer indices."""
        expanded = {}
        for omnilens_name, native_path in self._mapping.items():
            if "{i}" in omnilens_name:
                for i in range(n_layers):
                    expanded[omnilens_name.replace("{i}", str(i))] = (
                        native_path.replace("{i}", str(i))
                    )
            else:
                expanded[omnilens_name] = native_path
        return Registry(expanded)


def load_registry(source: str | Path | None) -> Registry | None:
    """Load a registry from a built-in architecture name or a YAML file path.

    Args:
        source: Either an architecture name (e.g. 'llama') to load from
            built-ins, or a path to a YAML file.

    Returns:
        A Registry, or None if the source wasn't found.
    """
    if source is None:
        return None

    source = str(source)

    # Check if it's a file path
    path = Path(source)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return _load_yaml(path)

    # Check built-in architectures
    builtin_path = ARCHITECTURES_DIR / f"{source}.yaml"
    if builtin_path.exists():
        return _load_yaml(builtin_path)

    return None


def save_registry(registry: Registry, path: str | Path) -> None:
    """Save a registry to a YAML file.

    Args:
        path: File path to save to (should end in .yaml or .yml).
    """
    path = Path(path)

    # Collapse expanded layers back to {i} templates where possible
    mapping = dict(registry.items())
    collapsed = _collapse_layers(mapping)

    data = {"mapping": collapsed}
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _collapse_layers(mapping: dict[str, str]) -> dict[str, str]:
    """Try to collapse numbered layer entries back to {i} templates."""
    import re

    templates: dict[str, str] = {}
    non_layer: dict[str, str] = {}

    for omnilens_name, native_path in mapping.items():
        # Check if this looks like a layer entry (has a number)
        match = re.match(r"^(layers\.)(\d+)(\..*)", omnilens_name)
        if match:
            template_key = f"{match.group(1)}{{i}}{match.group(3)}"
            native_template = re.sub(
                r"(\.)(\d+)(\.|$)",
                lambda m: f".{{i}}{m.group(3)}",
                native_path,
                count=1,
            )
            templates[template_key] = native_template
        else:
            non_layer[omnilens_name] = native_path

    result = {}
    result.update(non_layer)
    result.update(templates)
    return result


def _load_yaml(path: Path) -> Registry:
    """Load a registry from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    mapping = data.get("mapping", {})
    n_layers = data.get("n_layers", None)

    registry = Registry(mapping)
    if n_layers is not None:
        registry = registry.expand_layers(n_layers)
    return registry
