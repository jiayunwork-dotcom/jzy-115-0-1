"""工作点求根模块（括号法 / 二分法）。

求解的方程：

    f(Q) = p_fan(Q) - dp_duct(Q) = 0

* 风管压升 dp_duct(Q) = K Q^2 对 Q 严格单调递增（Q >= 0）；
* 风机特性通常随流量增加而下降；
* f(0) = 风机零流量全压（即关断/静压点，在同一截面上全压=静压，
  因为动压为 0）。

因此只要 f(0) >= 0 且大流量处 f 为负，工作点在
[0, +inf) 上唯一。采用括号法：从曲线可求值域起点出发，
几何扩张右括号直至出现异号，再二分收敛。曲线声明 reject
外推时，若交点落在采样范围之外会得到明确的错误而不是糊弄的数值。
"""

from __future__ import annotations

from dataclasses import dataclass

from .duct import Duct, resistance_pressure, velocity
from .fan_curve import (
    FanCurve,
    FanCurveOutOfRange,
    SampledFanCurve,
)


class SolverError(RuntimeError):
    """无法确定稳定工作点（无交点、括号搜索失败等）。"""


@dataclass(frozen=True)
class OperatingPoint:
    """风机-风管系统的稳定工作点。"""

    flow_m3s: float        # 交点体积流量 Q，m^3/s
    total_pressure_pa: float  # 工作点全压（= 风机全压 = 风管压升），Pa
    velocity_ms: float     # 该流量下管内流速，m/s（唯一口径 Q/A）
    duct_pressure_pa: float   # 风管需要的压升，Pa（应与全压一致）
    iterations: int        # 二分迭代次数
    fan_shutoff_pressure_pa: float | None  # Q=0 处风机全压（静压点），Pa


def _residual(fan: FanCurve, duct: Duct, flow_m3s: float) -> float:
    """f(Q) = 风机全压 - 风管压升。"""
    return fan.evaluate(flow_m3s) - resistance_pressure(duct, flow_m3s)


def _safe_residual(
    fan: FanCurve, duct: Duct, flow_m3s: float
) -> tuple[bool, float]:
    """求值并把"曲线拒绝外推"翻译成括号法可识别的信号。

    返回 (是否可求值, f(Q))。
    """
    try:
        return True, _residual(fan, duct, flow_m3s)
    except FanCurveOutOfRange:
        return False, 0.0


def solve_operating_point(
    fan: FanCurve,
    duct: Duct,
    *,
    flow_tol_m3s: float = 1.0e-9,
    max_bisections: int = 80,
) -> OperatingPoint:
    """求风机特性曲线与风管阻力曲线的稳定交点。

    抛出
    ----
    FanCurveOutOfRange
        曲线声明拒绝外推，而交点落在其声明范围之外。
    SolverError
        物理上不存在工作点（零流量处风机已压不过风管；零流量时
        风管压升为 0，等价于风机关断全压为负）或搜索未能闭合。
    """
    q_low = fan.domain_start()
    ok_low, f_low = _safe_residual(fan, duct, q_low)
    if not ok_low:  # 理论上 domain_start 必须可求值，双保险。
        raise SolverError("风机曲线在其声明域起点处不可求值")

    # Q = 0 时风管压升为 0，f(0) 即风机的关断全压（静压点）。
    # 若它为负，系统在正流量上不可能有交点，明确报"无工作点"。
    if q_low == 0.0:
        shutoff: float | None = f_low
    else:
        ok0, f0 = _safe_residual(fan, duct, 0.0)
        shutoff = f0 if ok0 else None
    if shutoff is not None and shutoff < 0.0:
        raise SolverError(
            "风机零流量全压为负，无法克服风管（Q=0 时管阻为 0），"
            "系统不存在正流量稳定工作点"
        )

    if f_low < 0.0:
        # 仅 reject 的采样曲线会走到这里（domain_start 是最小采样点）：
        # 起点处 f<0 说明交点在采样范围的更小流量侧之外。
        raise FanCurveOutOfRange(
            f"工作点位于风机曲线声明范围的起点 {q_low:g} m^3/s 之外（更小流量侧），"
            "按声明策略拒绝外推"
        )

    # 可求值流量域的右端：reject 的采样曲线止于最后一个采样点，
    # 其余情形止于声明的 max_flow_m3s。
    if isinstance(fan, SampledFanCurve) and fan.extrapolation == "reject":
        domain_end = fan.max_sampled_flow
    else:
        domain_end = fan.max_flow_m3s

    # 几何扩张右括号：0.01, 0.02, 0.04, ...，但绝不越过可求值域右端。
    q_high = max(min(q_low + 1.0e-2, domain_end), 1.0e-2)
    expansions = 0
    while True:
        q_high = min(q_high, domain_end)
        ok_high, f_high = _safe_residual(fan, duct, q_high)
        if (not ok_high) or f_high <= 0.0:
            break  # 撞上不可求值边界，或括号已闭合
        if q_high >= domain_end:
            break  # 已到声明域右端且余量仍为正 -> 交点在更大流量侧之外
        q_high *= 2.0
        expansions += 1
        if expansions > 200:
            raise SolverError("括号搜索未能闭合：扩张步数超限，请检查风机曲线")

    # 精确落在左括号端点上。
    if f_low == 0.0:
        q_star, iters = q_low, 0
    elif not ok_high or f_high > 0.0:
        # 右端不可求值（linear 外推撞上上限），或右端余量仍为正
        # （reject 采样曲线最后一点仍压得过风管）：都说明交点不在
        # 事先声明的可求值范围内，必须明确报错，不能默默给数。
        raise FanCurveOutOfRange(
            "工作点位于风机曲线声明范围之外（更大流量侧），按声明策略拒绝外推"
        )
    elif f_high == 0.0:
        q_star, iters = q_high, expansions
    else:
        # 二分法。这里采用经典括号求根，而不是直接套解析公式，
        # 以适用于任意形状的采样/拟合曲线。
        iters = expansions
        for _ in range(max_bisections):
            q_mid = 0.5 * (q_low + q_high)
            f_mid = _residual(fan, duct, q_mid)
            iters += 1
            if f_mid > 0.0:
                q_low = q_mid
            else:
                q_high = q_mid
            if (q_high - q_low) <= flow_tol_m3s or f_mid == 0.0:
                break
        q_star = 0.5 * (q_low + q_high)

    p_duct = resistance_pressure(duct, q_star)
    p_fan = fan.evaluate(q_star)
    residual = p_fan - p_duct
    # 回代后的交点必须真的"立得住"：风机全压 ≈ 风管压升。
    scale = max(1.0, abs(p_fan), abs(p_duct))
    if abs(residual) > 1.0e-6 * scale:
        raise SolverError(
            f"求根结果未闭合: |风机全压 - 风管压升| = {abs(residual):.3g} Pa"
        )

    return OperatingPoint(
        flow_m3s=q_star,
        total_pressure_pa=p_fan,
        velocity_ms=velocity(duct, q_star),
        duct_pressure_pa=p_duct,
        iterations=iters,
        fan_shutoff_pressure_pa=shutoff,
    )
