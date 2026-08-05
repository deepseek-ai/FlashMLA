"""Runtime SM/grid configuration for the production CSA pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import os
from pathlib import Path
from typing import Any, Mapping


SM_CONFIG_ENV = "FLASH_MLA_CSA_SM_CONFIG"
ROOT_GRID_SMS = 148
ROOT_PROJECTION_CHOICES = tuple(range(96, 110, 2))
ROOT_GB200_GRID_SMS = 152
ROOT_GB200_PROJECTION_CHOICES = (104, 106, 108)


@dataclass(frozen=True, slots=True)
class ProductionSMConfig:
    """All launch-time SM budgets used by ROOT + K1--K13.

    Every field is a runtime launch parameter.  ROOT has accepted 148-CTA B200
    and 152-CTA GB200 cluster launches; its internal projection/index split is
    selected from precompiled specializations so changing it never rebuilds or
    JIT-compiles the extension.
    """

    root_grid_sms: int = ROOT_GRID_SMS
    root_projection_sms: int = 104
    state_save_sms: int = 96
    main_compress_sms: int = 96
    swa_cache_insert_sms: int = 4
    index_query_sms: int = 56
    index_cache_sms: int = 148
    index_logits_sms: int = 148
    topk_sms: int = 148
    kv_cache_gather_sms: int = 4
    combine_sms: int = 148
    reset_sms: int = 148
    wqb_launch_sms: int = 148
    wqb_steady_sms: int = 36
    prefix_attention_sms: int = 108
    tail_attention_sms: int = 64
    tail_output_sms: int = 64
    prefix_output_sms: int = 84
    output_quant_sms: int = 148
    output_projection_sms: int = 148

    @property
    def root_index_sms(self) -> int:
        return self.root_grid_sms - self.root_projection_sms

    def as_dict(self) -> dict[str, int]:
        result = asdict(self)
        result["root_index_sms"] = self.root_index_sms
        return result

    def validate(self, physical_sms: int) -> "ProductionSMConfig":
        if physical_sms not in (148, 152):
            raise ValueError(
                "production SM configuration requires a 148-SM B200 or "
                f"152-SM GB200-class device, got {physical_sms} SMs"
            )
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field.name} must be an integer")
            if not 0 < value <= physical_sms:
                raise ValueError(
                    f"{field.name}={value} must be in [1,{physical_sms}]"
                )
        if self.root_grid_sms == ROOT_GRID_SMS:
            choices = ROOT_PROJECTION_CHOICES
        elif self.root_grid_sms == ROOT_GB200_GRID_SMS:
            choices = ROOT_GB200_PROJECTION_CHOICES
        else:
            raise ValueError(
                "root_grid_sms must select a precompiled total from {148,152}"
            )
        if self.root_projection_sms not in choices:
            choices_text = ", ".join(str(value) for value in choices)
            raise ValueError(
                "root_projection_sms must select a precompiled specialization "
                f"for grid{self.root_grid_sms} from {{{choices_text}}}"
            )
        if self.root_index_sms <= 0 or self.root_index_sms % 2:
            raise ValueError("root index lane must contain a positive even CTA count")
        for name in (
            "state_save_sms",
            "index_query_sms",
            "index_logits_sms",
            "wqb_launch_sms",
            "wqb_steady_sms",
            "prefix_attention_sms",
            "tail_attention_sms",
            "tail_output_sms",
            "prefix_output_sms",
            "output_projection_sms",
        ):
            if getattr(self, name) % 2:
                raise ValueError(f"{name} must be even for its cluster-2 kernel")
        if not self.wqb_steady_sms < self.wqb_launch_sms:
            raise ValueError("wqb_steady_sms must be smaller than wqb_launch_sms")
        return self

    @classmethod
    def for_device(cls, physical_sms: int) -> "ProductionSMConfig":
        if physical_sms == 148:
            return cls().validate(physical_sms)
        if physical_sms == 152:
            return cls(
                root_grid_sms=152,
                root_projection_sms=108,
                combine_sms=152,
                reset_sms=152,
                wqb_launch_sms=152,
                wqb_steady_sms=38,
                prefix_attention_sms=152,
                tail_attention_sms=68,
                tail_output_sms=68,
                output_quant_sms=152,
                output_projection_sms=152,
            ).validate(physical_sms)
        return cls().validate(physical_sms)

    def with_overrides(
        self, overrides: Mapping[str, Any], *, physical_sms: int
    ) -> "ProductionSMConfig":
        valid = {field.name for field in fields(self)}
        unknown = sorted(set(overrides).difference(valid))
        if unknown:
            raise ValueError("unknown SM configuration keys: " + ", ".join(unknown))
        return replace(self, **dict(overrides)).validate(physical_sms)


def _parse_external(value: str) -> Mapping[str, Any]:
    source = value.strip()
    if source.startswith("@"):
        source = Path(source[1:]).read_text()
    elif source and source[0] != "{":
        candidate = Path(source)
        if candidate.is_file():
            source = candidate.read_text()
    payload = json.loads(source)
    if not isinstance(payload, dict):
        raise ValueError("SM configuration JSON must be an object")
    return payload


def resolve_sm_config(
    physical_sms: int,
    config: ProductionSMConfig | Mapping[str, Any] | str | None = None,
) -> ProductionSMConfig:
    """Resolve defaults plus an object, mapping, JSON string/path, or env value."""

    if config is None:
        config = os.environ.get(SM_CONFIG_ENV)
    if isinstance(config, ProductionSMConfig):
        return config.validate(physical_sms)
    base = ProductionSMConfig.for_device(physical_sms)
    if config is None or config == "":
        return base
    if isinstance(config, str):
        config = _parse_external(config)
    if not isinstance(config, Mapping):
        raise TypeError("SM configuration must be a mapping, JSON, or ProductionSMConfig")
    return base.with_overrides(config, physical_sms=physical_sms)


__all__ = [
    "ProductionSMConfig",
    "ROOT_GRID_SMS",
    "ROOT_PROJECTION_CHOICES",
    "ROOT_GB200_GRID_SMS",
    "ROOT_GB200_PROJECTION_CHOICES",
    "SM_CONFIG_ENV",
    "resolve_sm_config",
]
