"""风机特性曲线模块（纯计算，不依赖 Web 框架）。

支持两种输入：

1. 若干组 (流量, 全压[, 轴功率]) 采样点 —— 点间线性插值；
   流量落在采样范围之外时按构造时声明的策略处理：
   * ``"reject"``（默认，安全）：直接抛 :class:`ExtrapolationError`，
     绝不默默给个数；
   * ``"linear"``：按端点处的线性斜率外推（事先声明的唯一外推方式）。
2. 拟合好的二次曲线 p(Q) = a2·Q² + a1·Q + a0，多项式全局成立，
   不存在"采样范围"概念（Q=0 时 p = a0，即风机静压/关闭压）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from .errors import ExtrapolationError, ValidationError

ExtrapolationMode = Literal["reject", "linear"]

_EXTRAPOLATION_MODES = ("reject", "linear")


@dataclass(frozen=True, slots=True)
class FanPoint:
    """一个风机特性采样点。"""

    flow_m3s: float
    total_pressure_pa: float
    shaft_power_w: float | None = None


# 可接受的采样点原始形式：FanPoint、(Q, p) 或 (Q, p, 轴功率|None)
RawPoint = (
    FanPoint
    | tuple[float, float]
    | tuple[float, float, float | None]
)


@dataclass(frozen=True, slots=True)
class FanCurve:
    """风机全压-流量特性曲线。

    两种构造方式互斥：

    * :meth:`from_samples`：采样点插值曲线；
    * :meth:`from_quadratic`：二次拟合曲线 p(Q) = a2 Q² + a1 Q + a0。
    """

    points: tuple[FanPoint, ...] | None = None
    a2: float = 0.0
    a1: float = 0.0
    a0: float = 0.0
    extrapolation: str = "reject"
    is_quadratic: bool = False

    # ---- 构造 ----------------------------------------------------------

    @staticmethod
    def from_samples(
        points: list[RawPoint],
        extrapolation: ExtrapolationMode = "reject",
    ) -> FanCurve:
        if extrapolation not in _EXTRAPOLATION_MODES:
            raise ValidationError(
                f"外推模式必须是 {_EXTRAPOLATION_MODES} 之一，收到 {extrapolation!r}"
            )
        if points is None or len(points) == 0:
            raise ValidationError("风机曲线采样点不能为空")
        if len(points) < 2:
            raise ValidationError("风机曲线采样点至少需要 2 个才能插值")

        normalized: list[FanPoint] = []
        for i, p in enumerate(points):
            if isinstance(p, FanPoint):
                fp = p
            else:
                if len(p) == 2:
                    fp = FanPoint(float(p[0]), float(p[1]))
                elif len(p) == 3:
                    fp = FanPoint(float(p[0]), float(p[1]), None if p[2] is None else float(p[2]))
                else:
                    raise ValidationError(f"第 {i} 个采样点必须是 (流量, 全压[, 轴功率])")
            if any(
                not isinstance(v, (int, float)) or isinstance(v, bool)
                for v in (fp.flow_m3s, fp.total_pressure_pa)
            ):
                raise ValidationError(f"第 {i} 个采样点的流量与全压必须是数值")
            if fp.shaft_power_w is not None and (
                not isinstance(fp.shaft_power_w, (int, float))
                or isinstance(fp.shaft_power_w, bool)
            ):
                raise ValidationError(f"第 {i} 个采样点的轴功率必须是数值")
            if math.isnan(fp.flow_m3s) or math.isinf(fp.flow_m3s):
                raise ValidationError(f"第 {i} 个采样点流量非法")
            if math.isnan(fp.total_pressure_pa) or math.isinf(fp.total_pressure_pa):
                raise ValidationError(f"第 {i} 个采样点全压非法")
            if fp.flow_m3s < 0:
                raise ValidationError(f"第 {i} 个采样点流量不能为负：{fp.flow_m3s}")
            normalized.append(fp)

        normalized.sort(key=lambda p: p.flow_m3s)
        for prev, cur in pairwise(normalized):
            if cur.flow_m3s == prev.flow_m3s:
                raise ValidationError(f"采样点流量必须互不相同，重复值 {cur.flow_m3s}")

        return FanCurve(points=tuple(normalized), extrapolation=extrapolation)

    @staticmethod
    def from_quadratic(a2: float, a1: float, a0: float) -> FanCurve:
        coeffs = (a2, a1, a0)
        if any(not isinstance(c, (int, float)) or isinstance(c, bool) for c in coeffs):
            raise ValidationError("二次曲线系数必须是数值")
        if any(math.isnan(c) or math.isinf(c) for c in coeffs):
            raise ValidationError("二次曲线系数不能为 NaN 或无穷")
        return FanCurve(a2=float(a2), a1=float(a1), a0=float(a0), is_quadratic=True)

    # ---- 域信息 --------------------------------------------------------

    @property
    def q_min(self) -> float:
        if self.is_quadratic:
            return 0.0
        assert self.points is not None
        return self.points[0].flow_m3s

    @property
    def q_max(self) -> float:
        if self.is_quadratic:
            return math.inf
        assert self.points is not None
        return self.points[-1].flow_m3s

    # ---- 求值 ----------------------------------------------------------

    def pressure_pa(self, q: float) -> float:
        """给定体积流量 Q，返回风机能提供的全压 (Pa)。"""
        if q < 0:
            raise ValidationError(f"体积流量不能为负，收到 {q}")
        if self.is_quadratic:
            return self.a2 * q * q + self.a1 * q + self.a0
        return _piecewise_linear(self._q_p(), q, self.extrapolation)

    def static_shutoff_pressure_pa(self) -> float:
        """Q=0 处风机给出的全压（零流量时即风机静压/关闭压）。

        采样曲线若不含 Q=0 点：``reject`` 模式直接拒绝；
        ``linear`` 模式按端点线性外推到 Q=0。
        """
        return self.pressure_pa(0.0)

    def shaft_power_w(self, q: float) -> float | None:
        """给定 Q 返回轴功率 (W)；未提供功率采样（或二次曲线未携带）时为 None。"""
        if q < 0:
            raise ValidationError(f"体积流量不能为负，收到 {q}")
        if self.is_quadratic or self.points is None:
            return None
        pairs: list[tuple[float, float]] = []
        for p in self.points:
            if p.shaft_power_w is not None:
                pairs.append((p.flow_m3s, p.shaft_power_w))
        if not pairs:
            return None
        if len(pairs) == 1:
            return pairs[0][1]
        return _piecewise_linear(pairs, q, self.extrapolation)

    # ---- 内部 ----------------------------------------------------------

    def _q_p(self) -> list[tuple[float, float]]:
        assert self.points is not None
        return [(p.flow_m3s, p.total_pressure_pa) for p in self.points]


def _piecewise_linear(
    pairs: list[tuple[float, float]], q: float, extrapolation: str
) -> float:
    """对 (x, y) 序列做分段线性插值；越界按声明策略拒绝或线性外推。"""
    x0, y0 = pairs[0]
    xn, yn = pairs[-1]

    if q < x0 or q > xn:
        if extrapolation == "reject":
            raise ExtrapolationError(
                f"流量 {q:.6g} m³/s 超出风机采样范围 [{x0:.6g}, {xn:.6g}]，"
                "且曲线声明 extrapolation=reject"
            )
        if q < x0:
            x1, y1 = pairs[1]
            t = (q - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
        x_prev, y_prev = pairs[-2]
        t = (q - xn) / (xn - x_prev)
        return yn + t * (yn - y_prev)

    for (xa, ya), (xb, yb) in pairwise(pairs):
        if xa <= q <= xb:
            if xb == xa:
                return ya
            t = (q - xa) / (xb - xa)
            return ya + t * (yb - ya)
    return yn
