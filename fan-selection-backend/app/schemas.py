"""HTTP 层的 Pydantic 请求/响应模型（schema 与核心逻辑分离）。

分层约定：Pydantic 只负责 JSON 结构、字段类型与数值有限性；
物理/数值范围（管径为正、密度为正、转速为正……）全部交给
``app.validation``，使非法输入的错误说明统一为中文。

字段名与单位都写在描述里，自动生成到 /docs 的 OpenAPI 文档中。
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

from .duct import AIR_DENSITY_20C


def _finite(name: str):  # type: ignore[no-untyped-def]
    """生成一个拒绝 NaN/无穷大的字段校验器。"""

    @field_validator(name)
    @classmethod
    def _check(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"{name} 必须是有限数值，不能是 NaN 或无穷大")
        return v

    return _check


# ----------------------- 请求模型 -----------------------


class DuctInput(BaseModel):
    length_m: float = Field(..., description="管长 L，m，必须 >= 0；负值在求解前被拦截")
    diameter_m: float = Field(..., description="圆管内径 D，m，必须 > 0；零/负值在求解前被拦截")
    friction_factor: float = Field(0.0, description="沿程（达西）摩阻系数 lambda，无量纲，>= 0")
    local_loss_coeff: float = Field(0.0, description="局部阻力系数之和 sum(zeta)，无量纲，>= 0")
    density: float = Field(
        AIR_DENSITY_20C,
        description="空气密度 rho，kg/m^3，必须为正；默认 1.2（20 ℃空气）",
    )

    _v_length = _finite("length_m")
    _v_diameter = _finite("diameter_m")
    _v_lambda = _finite("friction_factor")
    _v_zeta = _finite("local_loss_coeff")
    _v_density = _finite("density")


class SampledFanInput(BaseModel):
    kind: Literal["sampled"] = "sampled"
    flows_m3s: list[float] = Field(
        ..., description="流量采样点 Q，m^3/s，至少 2 个且严格递增（在求解前校验）"
    )
    pressures_pa: list[float] = Field(..., description="对应的全压采样点，Pa，数量与流量点一致")
    extrapolation: Literal["reject", "linear"] = Field(
        "reject",
        description="流量超出采样范围时的事先声明策略：reject=拒绝，linear=末段线性外推",
    )
    max_flow_m3s: float = Field(
        100.0, description="外推/求根允许到达的最大流量，m^3/s，必须为正"
    )

    @field_validator("flows_m3s", "pressures_pa")
    @classmethod
    def _all_finite(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(v) for v in values):
            raise ValueError("采样点数值必须有限，不能含 NaN 或无穷大")
        return values

    _v_max = _finite("max_flow_m3s")


class QuadraticFanInput(BaseModel):
    kind: Literal["quadratic"] = "quadratic"
    a2: float = Field(..., description="二次项系数，p = a2 Q^2 + a1 Q + a0（通常为负）")
    a1: float = Field(..., description="一次项系数")
    a0: float = Field(..., description="零流量全压（关断静压），Pa")
    max_flow_m3s: float = Field(100.0, description="求根允许到达的最大流量，m^3/s，必须为正")

    _v_a2 = _finite("a2")
    _v_a1 = _finite("a1")
    _v_a0 = _finite("a0")
    _v_max = _finite("max_flow_m3s")


FanInput = Annotated[
    Union[SampledFanInput, QuadraticFanInput],
    Field(discriminator="kind"),
]


class OperatingPointRequest(BaseModel):
    duct: DuctInput
    fan: FanInput


class AffinityRequest(BaseModel):
    duct: DuctInput
    fan: FanInput
    base_speed_rpm: float = Field(..., description="风机曲线对应的基准转速 n1，rpm，必须 > 0")
    new_speed_rpm: float = Field(..., description="目标转速 n2，rpm，必须 > 0")
    fan_total_efficiency: float | None = Field(
        None,
        description="可选：风机全压效率，(0, 1]；给定后才输出轴功率"
                    "（假设相似工况下近似不变）",
    )

    _v_n1 = _finite("base_speed_rpm")
    _v_n2 = _finite("new_speed_rpm")

    @field_validator("fan_total_efficiency")
    @classmethod
    def _eta_finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("效率必须是有限数值")
        return v


# ----------------------- 响应模型 -----------------------


class OperatingPointOutput(BaseModel):
    flow_m3s: float = Field(..., description="交点体积流量，m^3/s")
    total_pressure_pa: float = Field(..., description="工作点全压，Pa")
    duct_pressure_pa: float = Field(..., description="该流量下风管需要的压升，Pa（与全压闭合）")
    velocity_ms: float = Field(..., description="管内流速 v=Q/A，m/s，全项目唯一口径")
    iterations: int = Field(..., description="二分求根迭代次数")
    fan_shutoff_pressure_pa: float | None = Field(
        None, description="Q=0 时风机全压（静压点），Pa"
    )


class PowerOutput(BaseModel):
    old_w: float
    new_w: float
    ratio: float = Field(..., description="新/旧功率比，应等于转速比的三次方")


class AffinityOutput(BaseModel):
    speed_ratio: float
    re_solved: bool = Field(
        ..., description="恒为 true：新工作点由变换后的曲线重新求交得到"
    )
    old: OperatingPointOutput
    new: OperatingPointOutput
    flow_ratio: float = Field(..., description="Q2/Q1，应等于转速比")
    pressure_ratio: float = Field(..., description="p2/p1，应等于转速比平方")
    air_power: PowerOutput = Field(..., description="空气功率 Qp，W")
    shaft_power: PowerOutput | None = Field(
        None, description="轴功率 Qp/eta，W；未给效率时为空"
    )


class ErrorResponse(BaseModel):
    error: str = Field(..., description="错误类别代码")
    detail: str = Field(..., description="中文说明")
