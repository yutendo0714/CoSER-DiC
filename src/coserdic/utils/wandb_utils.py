from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def init_wandb(config: Mapping[str, Any], run_name: str | None = None):
    """Initialize W&B lazily so tests do not import it unless needed."""

    import wandb

    wandb_cfg = dict(config.get("wandb", {}))
    project = wandb_cfg.pop("project", "coserdic")
    entity = wandb_cfg.pop("entity", None)
    mode = wandb_cfg.pop("mode", None)
    tags = wandb_cfg.pop("tags", None)
    return wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        tags=tags,
        name=run_name,
        config=dict(config),
        **wandb_cfg,
    )

