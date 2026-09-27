"""HTTP 路由层：只负责 序列化/反序列化 与调用领域模块，不写物理/求根逻辑。"""

from __future__ import annotations

from fastapi import APIRouter

from .affinity import solve_at_speed
from .duct import Duct
from .fan import FanCurve, FanPoint
from .schemas import (
    DuctIn,
    FanCurveIn,
    OperatingPointOut,
    OperatingPointRequest,
    SpeedRequest,
    SpeedResponse,
)
from .solver import OperatingPoint, solve_operating_point

router = APIRouter()


def build_duct(d: DuctIn) -> Duct:
    return Duct(
        length_m=d.length_m,
        diameter_m=d.diameter_m,
        friction_factor=d.friction_factor,
        local_loss_sum=d.local_loss_sum,
        density=d.density,
    )


def build_fan(f: FanCurveIn) -> FanCurve:
    if f.samples is not None:
        points = [
            FanPoint(
                flow_m3s=s.flow_m3s,
                total_pressure_pa=s.total_pressure_pa,
                shaft_power_w=s.shaft_power_w,
            )
            for s in f.samples
        ]
        return FanCurve.from_samples(points, extrapolation=f.extrapolation)
    assert f.a2 is not None and f.a1 is not None and f.a0 is not None
    return FanCurve.from_quadratic(f.a2, f.a1, f.a0)


def _point_to_out(point: OperatingPoint, duct: Duct) -> OperatingPointOut:
    duct_p = duct.required_pressure(point.flow_m3s)
    return OperatingPointOut(
        flow_m3s=point.flow_m3s,
        total_pressure_pa=point.total_pressure_pa,
        velocity_ms=point.velocity_ms,
        dynamic_pressure_pa=duct.dynamic_pressure(point.flow_m3s),
        duct_required_pressure_pa=duct_p,
        residual_pa=point.total_pressure_pa - duct_p,
        shaft_power_w=point.shaft_power_w,
        fan_static_pressure_pa=point.fan_static_pressure_pa,
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/operating-point", response_model=OperatingPointOut)
def operating_point(req: OperatingPointRequest) -> OperatingPointOut:
    """给定一段风管与一台风机曲线，求真正稳定运行的工作点。"""
    duct = build_duct(req.duct)
    fan = build_fan(req.fan)
    point = solve_operating_point(fan, duct)
    return _point_to_out(point, duct)


@router.post("/speed", response_model=SpeedResponse)
def speed(req: SpeedRequest) -> SpeedResponse:
    """指定新转速：按相似律整体变换风机曲线后，与原风管重新求交。"""
    duct = build_duct(req.duct)
    fan = build_fan(req.fan)
    sol = solve_at_speed(
        fan, duct, req.base_speed_rpm, req.new_speed_rpm
    )
    return SpeedResponse(
        base_speed_rpm=req.base_speed_rpm,
        new_speed_rpm=req.new_speed_rpm,
        speed_ratio=sol.speed_ratio,
        base=_point_to_out(sol.base, duct),
        new=_point_to_out(sol.new, duct),
        flow_ratio=sol.flow_ratio,
        pressure_ratio=sol.pressure_ratio,
        power_ratio=sol.power_ratio,
    )
