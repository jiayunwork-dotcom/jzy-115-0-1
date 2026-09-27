"""风机相似律（affinity laws）模块。

同一台风机、同一系统、只改变转速 n 时（叶轮直径不变）::

    Q2 / Q1 = n2 / n1            （流量 ∝ 转速）
    p2 / p1 = (n2 / n1)^2        （全压 ∝ 转速²）
    P2 / P1 = (n2 / n1)^3        （轴功率 ∝ 转速³）

重要：转速一变，风机整条特性曲线都变了。本模块先按相似律把整条曲线
变换到新转速，再调用 :mod:`app.solver` 与**原风管**重新求交。
绝不允许把旧工作点流量直接乘个转速比就返回。
"""

from __future__ import annotations

from dataclasses import dataclass

from .duct import Duct
from .errors import ValidationError
from .fan import FanCurve, FanPoint
from .solver import OperatingPoint, solve_operating_point


@dataclass(frozen=True, slots=True)
class SpeedSolution:
    """指定新转速后的完整换算结果（新交点是重新求解得到的）。"""

    speed_ratio: float
    base: OperatingPoint
    new: OperatingPoint
    flow_ratio: float
    pressure_ratio: float
    power_ratio: float | None


def scale_fan_curve(
    fan: FanCurve, speed_ratio: float
) -> FanCurve:
    """按转速比 r = n_new/n_base 整体变换风机特性曲线。

    采样点：每个 (Q, p, P) 变为 (rQ, r²p, r³P)（功率若有）。
    二次曲线 p = a2 Q² + a1 Q + a0：新曲线系数 (a2, a1·r, a0·r²)。
    """
    if speed_ratio <= 0:
        raise ValidationError(f"转速比必须为正，收到 {speed_ratio}")

    if fan.is_quadratic:
        return FanCurve.from_quadratic(
            fan.a2,
            fan.a1 * speed_ratio,
            fan.a0 * speed_ratio**2,
        )

    assert fan.points is not None
    scaled: list[FanPoint] = []
    for pt in fan.points:
        scaled.append(
            FanPoint(
                flow_m3s=pt.flow_m3s * speed_ratio,
                total_pressure_pa=pt.total_pressure_pa * speed_ratio**2,
                shaft_power_w=(
                    pt.shaft_power_w * speed_ratio**3
                    if pt.shaft_power_w is not None
                    else None
                ),
            )
        )
    return FanCurve.from_samples(scaled, extrapolation=fan.extrapolation)


def solve_at_speed(
    fan: FanCurve,
    duct: Duct,
    base_speed_rpm: float,
    new_speed_rpm: float,
) -> SpeedSolution:
    """求基准转速工作点，并在按相似律变换后的曲线上重新求交。"""
    if base_speed_rpm <= 0:
        raise ValidationError(f"基准转速必须为正，收到 {base_speed_rpm}")
    if new_speed_rpm <= 0:
        raise ValidationError(f"新转速必须为正，收到 {new_speed_rpm}")

    r = new_speed_rpm / base_speed_rpm

    base_point = solve_operating_point(fan, duct)
    new_fan = scale_fan_curve(fan, r)
    # 注意：新交点必须用新曲线与原风管重新求解。
    new_point = solve_operating_point(new_fan, duct)

    power_ratio = None
    if base_point.shaft_power_w is not None and new_point.shaft_power_w is not None:
        power_ratio = new_point.shaft_power_w / base_point.shaft_power_w

    return SpeedSolution(
        speed_ratio=r,
        base=base_point,
        new=new_point,
        flow_ratio=new_point.flow_m3s / base_point.flow_m3s if base_point.flow_m3s > 0 else None,
        pressure_ratio=new_point.total_pressure_pa / base_point.total_pressure_pa
        if base_point.total_pressure_pa > 0
        else None,
        power_ratio=power_ratio,
    )
