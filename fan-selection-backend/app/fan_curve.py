"""风机特性曲线求值模块。

支持两种事先声明的曲线形式：

1. 采样点曲线 ``SampledFanCurve``：
   若干组 (流量 Q, 全压 p) 采样点，按流量升序排列；
   采样点之间做线性插值。流量落在采样范围之外时如何处理必须事先声明：

   * ``reject``：明确拒绝（抛出 :class:`FanCurveOutOfRange`），
     绝不默默给数；
   * ``linear``：按边界处的最末一段做线性外推（声明过的外推方式）。

2. 二次曲线 ``QuadraticFanCurve``：
   p(Q) = a2 Q^2 + a1 Q + a0，系数在构造时给定。
   二次曲线在 Q >= 0 全域可求值；Q < 0 一律拒绝（物理上不允许倒流）。

两种曲线都暴露 :meth:`domain_start`（可求值流量域的起点），
供求根模块确定括号搜索的起点。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal, Union

# 外推策略：reject=超范围拒绝；linear=边界段线性外推。
ExtrapolationKind = Literal["reject", "linear"]


class FanCurveError(ValueError):
    """风机曲线本身或求值方式有问题。"""


class FanCurveOutOfRange(FanCurveError):
    """流量落在已声明的可求值范围之外（且策略为拒绝）。"""


@dataclass(frozen=True)
class SampledFanCurve:
    """由流量-全压采样点构成的风机特性曲线。"""

    flows_m3s: tuple[float, ...]
    pressures_pa: tuple[float, ...]
    extrapolation: ExtrapolationKind = "reject"
    # 外推搜索时允许到达的最大流量，防止线性外推无界狂奔。
    max_flow_m3s: float = 100.0

    def __post_init__(self) -> None:
        # 内容合法性（点数、排序、重复点）在 validation 模块也会校验，
        # 这里做一层结构性的最低保证，避免 dataclass 被直接误用。
        n = len(self.flows_m3s)
        if n < 2:
            raise FanCurveError("采样曲线至少需要 2 个采样点")
        if n != len(self.pressures_pa):
            raise FanCurveError("流量采样点与全压采样点数量必须一致")
        for prev, cur in zip(self.flows_m3s, self.flows_m3s[1:]):
            if cur <= prev:
                raise FanCurveError("采样点流量必须严格递增且不得重复")
        if self.extrapolation not in ("reject", "linear"):
            raise FanCurveError(f"未知的外推策略: {self.extrapolation!r}")

    @property
    def min_flow(self) -> float:
        return self.flows_m3s[0]

    @property
    def max_sampled_flow(self) -> float:
        return self.flows_m3s[-1]

    def domain_start(self) -> float:
        """reject 策略下只能从第一个采样点开始求值；linear 策略可从 0 起。"""
        return self.min_flow if self.extrapolation == "reject" else 0.0

    def evaluate(self, flow_m3s: float) -> float:
        """求风机在给定流量下能提供的全压，Pa。"""
        if flow_m3s < 0.0:
            raise FanCurveOutOfRange(f"流量不能为负: {flow_m3s!r}")

        qs, ps = self.flows_m3s, self.pressures_pa
        idx = bisect_left(qs, flow_m3s)

        # 恰好命中采样点。
        if idx < len(qs) and qs[idx] == flow_m3s:
            return ps[idx]

        # 落在采样区间内：线性插值。
        if 0 < idx < len(qs):
            q0, q1 = qs[idx - 1], qs[idx]
            p0, p1 = ps[idx - 1], ps[idx]
            return p0 + (p1 - p0) * (flow_m3s - q0) / (q1 - q0)

        # 低于最小采样流量。
        if idx == 0:
            if self.extrapolation == "reject":
                raise FanCurveOutOfRange(
                    f"流量 {flow_m3s:g} m^3/s 低于最小采样点 {qs[0]:g}，"
                    "曲线声明的策略是拒绝外推"
                )
            # 用首段线性外推。
            q0, q1, p0, p1 = qs[0], qs[1], ps[0], ps[1]
            return p0 + (p1 - p0) * (flow_m3s - q0) / (q1 - q0)

        # 高于最大采样流量。
        if flow_m3s > self.max_flow_m3s:
            raise FanCurveOutOfRange(
                f"外推流量 {flow_m3s:g} 超过允许上限 {self.max_flow_m3s:g}"
            )
        if self.extrapolation == "reject":
            raise FanCurveOutOfRange(
                f"流量 {flow_m3s:g} m^3/s 高于最大采样点 {qs[-1]:g}，"
                "曲线声明的策略是拒绝外推"
            )
        # 用末段线性外推。
        q0, q1, p0, p1 = qs[-2], qs[-1], ps[-2], ps[-1]
        return p0 + (p1 - p0) * (flow_m3s - q0) / (q1 - q0)

        raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class QuadraticFanCurve:
    """二次风机特性曲线 p(Q) = a2 Q^2 + a1 Q + a0。"""

    a2: float
    a1: float
    a0: float
    max_flow_m3s: float = 100.0

    def domain_start(self) -> float:
        # 多项式在全域有定义；物理域从 0 开始。
        return 0.0

    def evaluate(self, flow_m3s: float) -> float:
        if flow_m3s < 0.0:
            raise FanCurveOutOfRange(f"流量不能为负: {flow_m3s!r}")
        if flow_m3s > self.max_flow_m3s:
            raise FanCurveOutOfRange(
                f"流量 {flow_m3s:g} 超过允许上限 {self.max_flow_m3s:g}"
            )
        return self.a2 * flow_m3s**2 + self.a1 * flow_m3s + self.a0


FanCurve = Union[SampledFanCurve, QuadraticFanCurve]
