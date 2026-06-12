"""Base class for scene search constraints."""

from abc import ABC, abstractmethod

import numpy as np


class BaseConstraint(ABC):
    """Abstract base for scene filtering constraints.

    Each constraint defines:
    - UI components for parameter input (Gradio widgets)
    - A filter function that checks if a scene passes the constraint
    """
    name: str = ""
    description: str = ""

    @abstractmethod
    def get_params_spec(self) -> dict:
        """Return parameter specification: {param_name: {type, default, label, min, max, step}}.

        Used to dynamically build Gradio UI components.
        """

    @abstractmethod
    def filter(
        self,
        npz_path: str,
        npz_data: np.lib.npyio.NpzFile,
        params: dict,
        entry: dict | None = None,
    ) -> bool:
        """Return True if the scene passes this constraint.

        Args:
            npz_path: Path to the NPZ file.
            npz_data: Loaded NPZ data (numpy NpzFile).
            params: Dict of parameter values from UI.
            entry: Optional index entry dict (carries precomputed fields like
                replay ``metrics``). NPZ-only constraints can ignore it;
                metric constraints read fields from here instead of
                re-deriving them from the NPZ.
        """
