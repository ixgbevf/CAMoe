\
import json
from typing import Any, Dict
from pathlib import Path

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

def _sanitize_yaml_text(text: str) -> str:
    """Remove odd leading artifacts (BOM or a stray leading backslash line).

    Some environments introduce a lone '\\' as the very first line, which
    confuses YAML parsers. We strip a UTF-8 BOM and a first line that is just
    a backslash.
    """
    # Strip UTF-8 BOM if present
    if text and text[0] == '\ufeff':
        text = text.lstrip('\ufeff')
    # Remove a leading line that is exactly a single backslash
    if text.startswith('\\\n'):
        text = text[2:]
    # Handle Windows newlines as well
    if text.startswith('\\\r\n'):
        text = text[3:]
    # Also handle a case where first non-space token is a lone backslash on line 1
    lines = text.splitlines(True)
    if lines and lines[0].strip() == '\\':
        text = ''.join(lines[1:])
    return text

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML config. Please `pip install pyyaml`.")
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        text = _sanitize_yaml_text(text)
        return yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {p.suffix}")

def dump_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
