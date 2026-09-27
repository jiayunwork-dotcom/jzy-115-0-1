"""工作点求解模块：括号法（bisection）求风机曲线与管阻曲线的交点。

待求方程（一般非线性）::

    g(Q) = p_fan(Q) - p_duct(Q) = 0

物理可行域：Q >= 0，且 Q 不得落在风机曲线声明拒绝的区域之外。
求解步骤：

1. 在 Q=0 处取值：p_duct(0)=0，p_fan(0) 是风机静压，必须有 g(0) >= 0；
2. 从一个正流量起步，用"倍步法"向右扩张，直到找到 g(Q) <= 0 的点，
   形成 [g>=0, g<=0] 的括号；上界受风机采样域 q_max 限制；
3. 在括号上做二分（中点复用同一套 Q/A 流速与同一条风机曲线），
   直到括号宽度足够小；
4. 回代交点流量，得到工作点全压与管内流速。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .duct import Duct
from .errors import NoIntersectionError, SolverDomainError
from .fan import FanCurve

_XTOL_M3S = 1e-9
_XTOL_REL = 1e-10
_MAX_BISECTIONS = 100
_MAX_EXPANSIONS = 200
_FIRST_STEP_M3S = 1e-3


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """稳定运行工作点。"""

    flow_m3s: float
    total_pressure_pa: float
    velocity_ms: float
    shaft_power_w: float | None
    fan_static_pressure_pa: float


def solve_operating_point(
    fan: FanCurve,
    duct: Duct,
    *,
    flow_guess_m3s: float = _FIRST_STEP_M3S,
    xtol: float = _XTOL_M3S,
) -> OperatingPoint:
    """求 ``p_fan(Q) == p_duct(Q)`` 的稳定工作点。

    风机曲线在下降、管阻曲线随 Q² 上升时交点唯一且稳定。
    任何越界/无交点情况都会抛 :class:`SelectionError` 子类，
    不会返回一个糊弄过去的数。
    """

    def g(q: float) -> float:
        return fan.pressure_pa(q) - duct.required_pressure(q)

    # Q=0 评估：采样曲线若不含 Q=0 且声明 reject，这里会抛 ExtrapolationError，
    # 直接向上冒泡交给 HTTP 层（它是 SelectionError 的子类）。
    g0 = g(0.0)

    if g0 < 0:
        # Q=0 时风机全压连零管阻都撑不住（物理上不应出现于常规风机）。
        raise NoIntersectionError(
            f"Q=0 处风机全压低于管阻压升（残差 {g0:.6g} Pa），不存在正流量工作点"
        )
    if g0 == 0.0:
        q_star = 0.0
    else:
        q_star = _bisect_intersection(g, fan, flow_guess_m3s, xtol)

    p_fan = fan.pressure_pa(q_star)
    p_duct = duct.required_pressure(q_star)
    # 残差即两条曲线在交点的闭合误差。
    residual = abs(p_fan - p_duct)
    if residual > max(1.0, 1e-6 * max(1.0, abs(p_fan))):
        # 正常不会走到这里；走到说明括号/收敛出了问题，宁可报错也不糊弄。
        raise SolverDomainError(f"工作点残差过大：{residual:.6g} Pa")

    return OperatingPoint(
        flow_m3s=q_star,
        # 工作点全压取风机侧（与管阻侧在容差内一致）。
        total_pressure_pa=p_fan,
        velocity_ms=duct.velocity(q_star),
        shaft_power_w=fan.shaft_power_w(q_star),
        fan_static_pressure_pa=fan.static_shutoff_pressure_pa(),
    )


def _bisect_intersection(
    g, fan: FanCurve, first_step: float, xtol: float
) -> float:
    """先倍步找括号再二分。所有求值都被限制在风机曲线的可行域内。"""
    q_lo = 0.0
    g_lo = g(q_lo)
    if g_lo <= 0.0:
        return q_lo

    if first_step <= 0:
        raise SolverDomainError("求根初始步长必须为正")

    q_hi = first_step
    qmax = fan.q_max

    # 倍步扩张，直到右端点残差 <= 0；右端点不能越过风机采样上界。
    for _ in range(_MAX_EXPANSIONS):
        if q_hi >= qmax:
            q_hi = qmax
            g_hi = g(q_hi)
            if g_hi > 0.0:
                raise NoIntersectionError(
                    f"在风机采样上界 Q={qmax:.6g} m³/s 处风机全压仍高于管阻"
                    f"（残差 {g_hi:.6g} Pa），采样范围内不存在交点"
                )
            break
        g_hi = g(q_hi)
        if g_hi <= 0.0:
            break
        q_lo, g_lo = q_hi, g_hi
        q_hi *= 2.0
        if not math.isfinite(q_hi):
            raise SolverDomainError("扩张求根括号时流量溢出")
    else:  # pragma: no cover - 安全兜底
        raise SolverDomainError("找不到包围交点的流量区间")

    # 二分。
    for _ in range(_MAX_BISECTIONS):
        if q_hi - q_lo <= max(xtol, _XTOL_REL * max(1.0, q_hi)):
            break
        q_mid = 0.5 * (q_lo + q_hi)
        g_mid = g(q_mid)
        if g_mid > 0.0:
            q_lo = q_mid
        elif g_mid < 0.0:
            q_hi = q_mid
        else:
            return q_mid

    # 取残差更接近 0 的一侧作为交点流量。
    return q_lo if abs(g(q_lo)) <= abs(g(q_hi)) else q_hi
