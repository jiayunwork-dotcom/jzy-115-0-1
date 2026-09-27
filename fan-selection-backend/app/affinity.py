"""风机相似律（affinity laws）换算模块。

对同一台风机、同一系统、进口空气密度不变：

    Q2 / Q1 = n2 / n1          （流量 ∝ 转速）
    p2 / p1 = (n2 / n1)^2      （全压 ∝ 转速平方）
    P2 / P1 = (n2 / n1)^3      （轴功率 ∝ 转速立方）

关键纪律：转速一变，整条风机特性曲线都变。本模块先把曲线按
相似律整体变换（采样点/二次系数都随之变换），再调用求根器
**重新求解** 与同一根风管的交点，绝不能拿旧交点流量直接乘
转速比交差——只有在纯二次管阻（无静压提升）时两者数值才恰好
相等，一般系统中并不成立。

曲线整体变换规则（把每个采样点 (Q, p) 映射为 (r Q, r^2 p)）：
  * 二次曲线 p = a2 Q^2 + a1 Q + a0：由 p'(Q') = r^2 p(Q'/r) 恒等展开，
        p'(Q') = a2 Q'^2 + (r a1) Q' + r^2 a0
    即 a2 不变，a1 -> r a1，a0 -> r^2 a0。
  * 采样点：逐点 (Q, p) -> (r Q, r^2 p)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .duct import Duct
from .fan_curve import (
    FanCurve,
    QuadraticFanCurve,
    SampledFanCurve,
)
from .solver import OperatingPoint, solve_operating_point


@dataclass(frozen=True)
class AffinityResult:
    """相似律换算 + 重新求交的结果。"""

    speed_ratio: float
    old_point: OperatingPoint
    new_point: OperatingPoint
    flow_ratio: float       # Q2/Q1，应等于转速比
    pressure_ratio: float   # p2/p1，应等于转速比平方
    air_power_old_w: float  # 旧工况空气功率 Q*p，W
    air_power_new_w: float  # 新工况空气功率 Q*p，W（比值应等于 r^3）
    air_power_ratio: float
    shaft_power_old_w: float | None  # 旧工况轴功率 Qp/eta，W（给了效率才有）
    shaft_power_new_w: float | None
    shaft_power_ratio: float | None  # 应等于 r^3
    re_solved: bool  # 恒为 True：新交点来自重新求根，而非旧点乘比例


def scale_fan_curve(fan: FanCurve, speed_ratio: float) -> FanCurve:
    """按转速比 r 整体变换风机特性曲线。"""
    r = speed_ratio
    if isinstance(fan, SampledFanCurve):
        return SampledFanCurve(
            flows_m3s=tuple(r * q for q in fan.flows_m3s),
            pressures_pa=tuple(r * r * p for p in fan.pressures_pa),
            extrapolation=fan.extrapolation,
            max_flow_m3s=max(fan.max_flow_m3s, r * fan.max_flow_m3s),
        )
    if isinstance(fan, QuadraticFanCurve):
        return QuadraticFanCurve(
            a2=fan.a2,
            a1=r * fan.a1,
            a0=r * r * fan.a0,
            max_flow_m3s=max(fan.max_flow_m3s, r * fan.max_flow_m3s),
        )
    raise TypeError(f"不支持的风机曲线类型: {type(fan).__name__}")


def _air_power(op: OperatingPoint) -> float:
    """空气功率（有效功率）P_e = Q * p，W。"""
    return op.flow_m3s * op.total_pressure_pa


def apply_affinity(
    fan: FanCurve,
    duct: Duct,
    base_speed_rpm: float,
    new_speed_rpm: float,
    *,
    fan_total_efficiency: float | None = None,
    flow_tol_m3s: float = 1.0e-9,
) -> AffinityResult:
    """在原转速求交一次，再把曲线按相似律变换后重新求交。"""
    old_point = solve_operating_point(fan, duct, flow_tol_m3s=flow_tol_m3s)

    r = new_speed_rpm / base_speed_rpm
    scaled_fan = scale_fan_curve(fan, r)
    # 核心：对新曲线重新求根，而不是 old.flow * r。
    new_point = solve_operating_point(
        scaled_fan, duct, flow_tol_m3s=flow_tol_m3s
    )

    p_air_old = _air_power(old_point)
    p_air_new = _air_power(new_point)

    shaft_old = shaft_new = shaft_ratio = None
    if fan_total_efficiency is not None and fan_total_efficiency > 0.0:
        # 假定相似工况下全压效率近似不变。
        shaft_old = p_air_old / fan_total_efficiency
        shaft_new = p_air_new / fan_total_efficiency
        shaft_ratio = shaft_new / shaft_old

    return AffinityResult(
        speed_ratio=r,
        old_point=old_point,
        new_point=new_point,
        flow_ratio=new_point.flow_m3s / old_point.flow_m3s,
        pressure_ratio=new_point.total_pressure_pa
        / old_point.total_pressure_pa,
        air_power_old_w=p_air_old,
        air_power_new_w=p_air_new,
        air_power_ratio=p_air_new / p_air_old,
        shaft_power_old_w=shaft_old,
        shaft_power_new_w=shaft_new,
        shaft_power_ratio=shaft_ratio,
        re_solved=True,
    )
