"""风管阻力模块（纯物理计算，不依赖任何 Web 框架）。

模型（圆管、不可压缩、空气温度固定 20℃ 的物性在调用处传入）：

    A       = pi * D^2 / 4
    v       = Q / A                       # 全系统唯一的流速口径
    p_dyn   = rho * v^2 / 2
    dP_duct = (f * L / D + sum(zeta)) * p_dyn = K * p_dyn

K 对给定风管是常数，因此 dP_duct 正比于 Q^2，系数里只含 rho 与几何参数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ValidationError

# 20 ℃、标准大气压下的空气密度 (kg/m^3)。题目要求物性按 20℃ 固定。
AIR_DENSITY_20C = 1.204


@dataclass(frozen=True, slots=True)
class Duct:
    """一段圆管风管的几何参数与流体物性。

    Attributes:
        length_m: 管长 L (m)，必须 >= 0。
        diameter_m: 内径 D (m)，必须 > 0。
        friction_factor: 沿程摩阻系数 f（达西公式），必须 >= 0。
        local_loss_sum: 局部阻力系数之和 Σζ，必须 >= 0。
        density: 空气密度 ρ (kg/m^3)，默认取 20℃ 空气，必须 > 0。
    """

    length_m: float
    diameter_m: float
    friction_factor: float = 0.02
    local_loss_sum: float = 0.0
    density: float = AIR_DENSITY_20C

    def __post_init__(self) -> None:
        # 所有非法输入在任何物理计算发生之前就被挡下。
        if not isinstance(self.length_m, (int, float)) or isinstance(self.length_m, bool):
            raise ValidationError("管长必须是数值")
        if not isinstance(self.diameter_m, (int, float)) or isinstance(self.diameter_m, bool):
            raise ValidationError("管径必须是数值")
        if not isinstance(self.friction_factor, (int, float)) or isinstance(
            self.friction_factor, bool
        ):
            raise ValidationError("沿程摩阻系数必须是数值")
        if not isinstance(self.local_loss_sum, (int, float)) or isinstance(
            self.local_loss_sum, bool
        ):
            raise ValidationError("局部阻力系数之和必须是数值")
        if not isinstance(self.density, (int, float)) or isinstance(self.density, bool):
            raise ValidationError("空气密度必须是数值")

        if math.isnan(self.length_m) or math.isinf(self.length_m):
            raise ValidationError("管长不能为 NaN 或无穷")
        if math.isnan(self.diameter_m) or math.isinf(self.diameter_m):
            raise ValidationError("管径不能为 NaN 或无穷")
        if self.diameter_m <= 0:
            raise ValidationError(f"管径必须为正，收到 {self.diameter_m}")
        if self.length_m < 0:
            raise ValidationError(f"管长不能为负，收到 {self.length_m}")
        if self.friction_factor < 0:
            raise ValidationError(f"沿程摩阻系数不能为负，收到 {self.friction_factor}")
        if self.local_loss_sum < 0:
            raise ValidationError(f"局部阻力系数之和不能为负，收到 {self.local_loss_sum}")
        if math.isnan(self.density) or math.isinf(self.density):
            raise ValidationError("空气密度不能为 NaN 或无穷")
        if self.density <= 0:
            raise ValidationError(f"空气密度必须为正，收到 {self.density}")

    @property
    def area_m2(self) -> float:
        """圆管截面积 A = πD²/4 (m²)。"""
        return math.pi * self.diameter_m**2 / 4.0

    @property
    def loss_coefficient(self) -> float:
        """总损失系数 K = fL/D + Σζ（无量纲，常数）。"""
        return self.friction_factor * self.length_m / self.diameter_m + self.local_loss_sum

    def velocity(self, flow_m3s: float) -> float:
        """流速 v = Q/A —— 全系统计算动压时使用的唯一口径 (m/s)。"""
        if flow_m3s < 0:
            raise ValidationError(f"体积流量不能为负，收到 {flow_m3s}")
        return flow_m3s / self.area_m2

    def dynamic_pressure(self, flow_m3s: float) -> float:
        """动压 p_dyn = ρv²/2 (Pa)，v 只从 Q/A 得到。"""
        v = self.velocity(flow_m3s)
        return 0.5 * self.density * v * v

    def required_pressure(self, flow_m3s: float) -> float:
        """风管在给定体积流量下需要的压升 dP = K·p_dyn (Pa)。

        这就是单管阻力曲线，是 Q 的二次齐次函数：Q=0 时为 0。
        """
        if flow_m3s < 0:
            raise ValidationError(f"体积流量不能为负，收到 {flow_m3s}")
        return self.loss_coefficient * self.dynamic_pressure(flow_m3s)
