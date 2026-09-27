"""HTTP 层使用的 Pydantic 请求/响应模型与输入校验。

所有字段带约束：管径为零、管长为负、密度非正、曲线为空等非法输入
在这里（或领域对象构造时）就被挡下，绝不会流入求根过程。

单位约定（SI）：流量 m³/s，全压 Pa，流速 m/s，转速 rpm，轴功率 W。
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .duct import AIR_DENSITY_20C


class DuctIn(BaseModel):
    length_m: float = Field(..., ge=0, description="管长 L (m)，>= 0")
    diameter_m: float = Field(..., gt=0, description="圆管内径 D (m)，必须 > 0")
    friction_factor: float = Field(..., ge=0, description="沿程摩阻系数 f，>= 0")
    local_loss_sum: float = Field(
        0.0, ge=0, description="局部阻力系数之和 Σζ，>= 0"
    )
    # 物性按 20℃ 固定；允许显式传入密度以便验证密度变化的影响，但必须为正。
    density: float = Field(
        AIR_DENSITY_20C, gt=0, description="空气密度 ρ (kg/m³)，默认 20℃ 空气 1.204"
    )


class FanSampleIn(BaseModel):
    flow_m3s: float = Field(..., ge=0, description="采样点体积流量 (m³/s)")
    total_pressure_pa: float = Field(..., description="该流量下风机全压 (Pa)")
    shaft_power_w: float | None = Field(
        None, description="可选：该流量下轴功率 (W)，用于相似律功率核对"
    )


class FanCurveIn(BaseModel):
    """风机特性曲线：采样点 与 二次系数 二选一。"""

    samples: list[FanSampleIn] | None = Field(
        None, description="若干组 (流量, 全压[, 轴功率]) 采样点"
    )
    a2: float | None = Field(None, description="二次曲线 p=a2·Q²+a1·Q+a0 的 a2")
    a1: float | None = None
    a0: float | None = None
    extrapolation: Literal["reject", "linear"] = Field(
        "reject",
        description="采样范围外的处理：reject=明确拒绝（默认）；linear=端点线性外推",
    )

    @model_validator(mode="after")
    def _exactly_one_curve_form(self) -> FanCurveIn:
        has_samples = self.samples is not None and len(self.samples) > 0
        quad_fields = (self.a2, self.a1, self.a0)
        has_quadratic = any(c is not None for c in quad_fields)

        if not has_samples and not has_quadratic:
            raise ValueError("风机曲线不能为空：必须提供 samples 或二次系数 (a2,a1,a0)")
        if has_samples and has_quadratic:
            raise ValueError("采样点与二次系数不能同时提供，请二选一")
        if has_quadratic and not all(c is not None for c in quad_fields):
            raise ValueError("二次曲线必须同时给出 a2、a1、a0")
        if has_samples and len(self.samples) < 2:
            raise ValueError("采样点至少需要 2 个")
        return self

    @field_validator("samples")
    @classmethod
    def _validate_samples(cls, v: list[FanSampleIn] | None) -> list[FanSampleIn] | None:
        if v is None:
            return v
        flows = [s.flow_m3s for s in v]
        if any(math.isnan(f) or math.isinf(f) for f in flows):
            raise ValueError("采样点流量不能为 NaN 或无穷")
        if len(flows) != len(set(flows)):
            raise ValueError("采样点流量必须互不相同")
        return v


class OperatingPointRequest(BaseModel):
    duct: DuctIn
    fan: FanCurveIn
    speed_rpm: float | None = Field(
        None, gt=0, description="可选：输入曲线对应的转速（仅用于回显）"
    )


class SpeedRequest(BaseModel):
    duct: DuctIn
    fan: FanCurveIn
    base_speed_rpm: float = Field(..., gt=0, description="输入曲线对应的基准转速 (rpm)")
    new_speed_rpm: float = Field(..., gt=0, description="要换算到的新转速 (rpm)")


class OperatingPointOut(BaseModel):
    flow_m3s: float
    total_pressure_pa: float
    velocity_ms: float
    dynamic_pressure_pa: float
    duct_required_pressure_pa: float
    residual_pa: float
    shaft_power_w: float | None
    fan_static_pressure_pa: float


class SpeedResponse(BaseModel):
    base_speed_rpm: float
    new_speed_rpm: float
    speed_ratio: float
    base: OperatingPointOut
    new: OperatingPointOut
    flow_ratio: float | None
    pressure_ratio: float | None
    power_ratio: float | None
