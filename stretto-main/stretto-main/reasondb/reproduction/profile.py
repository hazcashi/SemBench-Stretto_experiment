"""One source of truth for model roles, routes and the experiment configuration."""
import json
from pathlib import Path

from reasondb.config.model_registry import ModelRegistry


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    registry = ModelRegistry.get()
    for role in ("small", "large"):
        spec = registry.spec_by_key(profile[role]["key"])
        if spec.modality != "text":
            raise ValueError("single-GPU profile supports text checkpoints only")
        profile[role] = {**profile[role], "model_name": spec.model_name,
                         "port": spec.port, "method_prefix": spec.method_prefix}
    if profile["small"]["port"] == profile["large"]["port"]:
        raise ValueError("model ports must be distinct")
    if profile["compression_ratio"] != 0.8 or profile["batch_size"] != 1:
        raise ValueError("v1 supports physical cr=0.8 and batch_size=1 only")
    if profile["reports_limit"] <= 0:
        raise ValueError("reports_limit must be positive")
    return profile


def build_configurator(profile):
    from reasondb.interface.default_operator_toolbox import OperatorSpec, build_toolbox
    from reasondb.optimizer.configurator import PlanConfigurator
    from reasondb.reasoning.llm import GPT4o

    small, large = (profile[r]["model_name"] for r in ("small", "large"))
    # These values express upstream's ordering assumptions, not measured accuracy.
    # Stretto's runtime/quality profiler observes the new models during optimization.
    specs = (
        OperatorSpec(small, .8, .3, 3.5, 1.0, 3.0),
        OperatorSpec(small, .0, .9, 5.0, 1.7, 3.7, vanilla=True),
        OperatorSpec(large, .8, .3, 6.5, 2.0, 4.0),
        OperatorSpec(large, .0, .9, 8.0, 2.9, 4.7, vanilla=True),
    )
    toolbox = build_toolbox(
        text_active=((small, .8), (large, .8)), image_active=(),
        text_specs=specs, image_specs=(), small_models=frozenset({small}),
        text_only=True, use_indexes=False, in_memory=(),
    )
    return PlanConfigurator(llm=GPT4o(), physical_operators=toolbox)
