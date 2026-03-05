"""Serializable configuration dataclasses for the guidance framework."""

from dataclasses import dataclass, field
import json


@dataclass
class GuidanceConfig:
    """Configuration for a single guidance function."""

    name: str
    enabled: bool = True
    scale: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class GuidanceSetConfig:
    """Full set of guidance functions for one experiment or inference call."""

    functions: list = field(default_factory=list)
    global_scale: float = 0.5

    def __post_init__(self):
        # Ensure elements are GuidanceConfig instances when deserialised from dict
        self.functions = [
            GuidanceConfig(**f) if isinstance(f, dict) else f
            for f in self.functions
        ]

    def to_json(self) -> str:
        import dataclasses
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "GuidanceSetConfig":
        data = json.loads(s)
        data["functions"] = [GuidanceConfig(**f) for f in data["functions"]]
        return cls(**data)

    @classmethod
    def from_file(cls, path: str) -> "GuidanceSetConfig":
        return cls.from_json(open(path).read())

    def save(self, path: str) -> None:
        open(path, "w").write(self.to_json())

    def active_functions(self) -> list:
        return [f for f in self.functions if f.enabled]
