"""CLI utilities for dataclass-based config. Works with any config class that marks fields with ``cli()``."""

import argparse
from dataclasses import MISSING, Field, field, fields
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

# Sentinel to detect "no default provided" vs "default=None".
_UNSET = object()


def cli(
    help: str,
    *,
    default: Any = _UNSET,
    default_factory: Any = _UNSET,
    path: bool = False,
) -> Any:
    """Mark a config field as exposed on the command line.

    - No default / default_factory  → field(default=MISSING)  → argparse required=True
    - default=None                 → field(default=None)      → argparse default=None
    - default=something            → field(default=something) → argparse default=something
    - default_factory=list         → field(factory=list)     → argparse default=[]
    """
    metadata = {"cli": True, "help": help, "path": path}
    if default is not _UNSET:
        return field(default=default, metadata=metadata)
    if default_factory is not _UNSET:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=MISSING, metadata=metadata)


def boolean(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def cli_fields(cls: type) -> list[Field]:
    return [f for f in fields(cls) if f.metadata.get("cli")]


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _add_argument(parser: argparse.ArgumentParser, f: Field) -> None:
    # Required only when NEITHER default nor default_factory is provided
    required = f.default is MISSING and f.default_factory is MISSING
    kwargs: dict[str, Any] = {"help": f.metadata["help"]}
    if required:
        kwargs["required"] = True
    elif f.default is not MISSING:
        kwargs["default"] = f.default
    elif f.default_factory is not MISSING:
        kwargs["default"] = f.default_factory()

    annotation = _unwrap_optional(f.type)
    if get_origin(annotation) is Literal:
        kwargs["type"] = str
        kwargs["choices"] = get_args(annotation)
    elif annotation is bool:
        kwargs["type"] = boolean
        kwargs["nargs"] = "?"
        kwargs["const"] = True
    elif get_origin(annotation) is list:
        kwargs.pop("type", None)
        kwargs["nargs"] = "+"
    else:
        kwargs["type"] = annotation

    parser.add_argument(f"--{f.name}", **kwargs)


def build_parser(cls: type, description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for f in cli_fields(cls):
        _add_argument(parser, f)
    return parser


def resolve_paths(args: argparse.Namespace, cls: type) -> None:
    for f in cli_fields(cls):
        if not f.metadata.get("path"):
            continue
        value = getattr(args, f.name)
        if not value:
            continue
        if isinstance(value, list):
            resolved = [str(Path(v).resolve()) for v in value]
            setattr(args, f.name, resolved)
        else:
            setattr(args, f.name, str(Path(value).resolve()))


def build_config(cls: type, args: argparse.Namespace, **overrides: Any) -> Any:
    values = {f.name: getattr(args, f.name) for f in cli_fields(cls)}
    values.update(overrides)
    return cls(**values)


def to_command_line(
    cfg_or_cls: Any, cls: type | None = None, exclude: tuple[str, ...] = ()
) -> list[str]:
    """Serialise a config back to argv.

    Two call shapes:
      - ``to_command_line(cfg, exclude=...)``  — cfg is a dataclass instance.
      - ``to_command_line(args, cls=TrainConfig, exclude=...)``  — args is an
        ``argparse.Namespace``; the matching dataclass must be passed via ``cls``
        because ``argparse.Namespace`` itself has no dataclass fields to inspect.
    """
    if cls is None:
        cls = type(cfg_or_cls)
        cfg = cfg_or_cls
    else:
        cfg = cfg_or_cls
    argv: list[str] = []
    for f in cli_fields(cls):
        if f.name in exclude:
            continue
        value = getattr(cfg, f.name)
        if value is None:
            continue
        if f.default is not MISSING and value == f.default:
            continue
        if isinstance(value, list):
            if not value:
                continue
            argv.append(f"--{f.name}")
            argv.extend(str(v) for v in value)
        else:
            argv += [f"--{f.name}", str(value)]
    return argv
