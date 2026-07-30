"""Resolve ``<run_id>@<epoch>`` model references to checkpoints and cache paths.

A model reference is the string passed to ``--model-id`` / ``--model-ids``. It is
either a bare training run id (``6a964bac``), which means the run's
``best_model.pt``, or a run id qualified with a checkpoint (``6a964bac@10``),
which means that run's ``epoch_010.pt``.

The qualifier accepts an epoch number (``@10``, ``@010``), an epoch filename with
or without extension (``@epoch_010``, ``@epoch_010.pt``), or any other checkpoint
stem in the run directory (``@last_model``).

Embedding caches are **suffixed** by the checkpoint rather than being written to a
separate directory, so the default (``best_model.pt``) keeps writing the historical
filenames untouched and no existing cache needs migrating::

    6a964bac      → cached_embeddings/resection_img_emb.parquet
    6a964bac@10   → cached_embeddings/resection_img_emb__ep010.parquet

The reference also supplies a filesystem-safe ``tag`` for naming output artifacts,
since ``@`` is awkward in shell globs and in result filenames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hcc_multimodal.eval.data import TRAINING_ROOT

DEFAULT_CHECKPOINT = "best_model.pt"
CACHE_SUBDIR = "cached_embeddings"


@dataclass(frozen=True)
class ModelRef:
    """A training run plus the checkpoint within it to read weights from."""

    spec: str  # as supplied on the command line, e.g. "6a964bac@10"
    run_id: str  # the training run directory name, e.g. "6a964bac"
    checkpoint: str  # checkpoint filename, e.g. "epoch_010.pt"
    suffix: str  # cache filename suffix, "" for the default checkpoint

    @property
    def run_dir(self) -> Path:
        return TRAINING_ROOT / self.run_id

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / self.checkpoint

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def cache_dir(self) -> Path:
        return self.run_dir / CACHE_SUBDIR

    @property
    def tag(self) -> str:
        """Filesystem-safe name for output dirs / result filenames."""
        return f"{self.run_id}{self.suffix}"

    def cache_path(self, stem: str) -> Path:
        """Cache file for an embedding ``stem`` (no extension), e.g. ``resection_img_emb``."""
        return self.cache_dir / f"{stem}{self.suffix}.parquet"

    def require_checkpoint(self) -> Path:
        """Return the checkpoint path, raising a helpful error if it is absent."""
        path = self.checkpoint_path
        if not path.exists():
            if not self.run_dir.is_dir():
                raise FileNotFoundError(f"No training run {self.run_id!r} at {self.run_dir}")
            available = sorted(p.name for p in self.run_dir.glob("*.pt"))
            raise FileNotFoundError(
                f"{self.spec!r} → no checkpoint {self.checkpoint!r} in {self.run_dir}. "
                f"Available: {', '.join(available) if available else '(none)'}"
            )
        return path


def parse_model_ref(spec: str) -> ModelRef:
    """Parse ``<run_id>[@<checkpoint>]`` into a :class:`ModelRef`.

    A bare run id resolves to ``best_model.pt`` with an empty cache suffix, so
    callers that never use the ``@`` syntax behave exactly as before.
    """
    run_id, sep, qualifier = spec.partition("@")
    if not run_id:
        raise ValueError(f"Model reference {spec!r} has an empty run id")
    if not sep:
        return ModelRef(spec=spec, run_id=run_id, checkpoint=DEFAULT_CHECKPOINT, suffix="")
    if not qualifier:
        raise ValueError(f"Model reference {spec!r} has an empty checkpoint qualifier")

    token = qualifier[:-3] if qualifier.endswith(".pt") else qualifier
    epoch_token = token[len("epoch_"):] if token.startswith("epoch_") else token
    if epoch_token.isdigit():
        epoch = int(epoch_token)
        return ModelRef(
            spec=spec,
            run_id=run_id,
            checkpoint=f"epoch_{epoch:03d}.pt",
            suffix=f"__ep{epoch:03d}",
        )
    # Any other checkpoint stem in the run dir, e.g. "@last_model".
    return ModelRef(spec=spec, run_id=run_id, checkpoint=f"{token}.pt", suffix=f"__{token}")
