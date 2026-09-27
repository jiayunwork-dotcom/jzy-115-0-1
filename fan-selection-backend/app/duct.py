"""风管阻力计算模块。

物理模型（定常、不可压缩、20 ℃空气）：

* 圆管截面积            A = pi * D^2 / 4
* 管内流速（唯一口径）  v = Q / A
* 动压                  pd = rho * v^2 / 2
* 沿程摩阻压损          dp_f = lambda * (L / D) * pd
* 局部阻力压损          dp_z = (sum zeta) * pd
* 风管需要的压升        dp = (lambda * L / D + sum zeta) * pd
                                  = K * Q^2 ，其中 K = (lambda*L/D + sum zeta) * rho / (2 A^2)

注意：全模块内流速只由 Q/A 这一条路径换算，管阻与任何下游计算
都不允许各自引入第二套流速。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 20 ℃、标准大气压附近的空气密度（工程常用值，kg/m^3）。
AIR_DENSITY_20C: float = 1.2


@dataclass(frozen=True)
class Duct:
    """一段圆形直风管的几何参数与阻力参数。

    属性
    ----
    length_m: 管长 L，m，必须 > 0。
    diameter_m: 圆管内径 D，m，必须 > 0。
    friction_factor: 沿程（达西）摩阻系数 lambda，无量纲，必须 >= 0。
    local_loss_coeff: 各局部阻力系数之和 sum(zeta)，无量纲，必须 >= 0。
    density: 空气密度 rho，kg/m^3，必须 > 0；默认取 20 ℃空气 1.2。
    """

    length_m: float
    diameter_m: float
    friction_factor: float = 0.0
    local_loss_coeff: float = 0.0
    density: float = AIR_DENSITY_20C

    @property
    def area_m2(self) -> float:
        """圆管截面积 A = pi D^2 / 4，m^2。"""
        return math.pi * self.diameter_m**2 / 4.0

    @property
    def total_loss_coeff(self) -> float:
        """总阻力系数 K_zeta = lambda L/D + sum(zeta)。"""
        return self.friction_factor * self.length_m / self.diameter_m + self.local_loss_coeff


def velocity(duct: Duct, flow_m3s: float) -> float:
    """由体积流量换算管内流速 v = Q / A，m/s。

    全项目流速只允许从这个函数取得，保证"同一个流量、同一个截面，
    流速只有一个口径"。
    """
    return flow_m3s / duct.area_m2


def dynamic_pressure(duct: Duct, flow_m3s: float) -> float:
    """动压 pd = rho v^2 / 2，Pa。"""
    v = velocity(duct, flow_m3s)
    return 0.5 * duct.density * v * v


def resistance_pressure(duct: Duct, flow_m3s: float) -> float:
    """风管在给定流量下需要的压升，Pa。

    dp = (lambda L/D + sum zeta) * pd，Q = 0 时恒为 0，
    且严格正比于 Q^2（截面、物性不变时）。
    """
    return duct.total_loss_coeff * dynamic_pressure(duct, flow_m3s)


def flow_squared_coefficient(duct: Duct) -> float:
    """管阻二次律系数 K，使 dp_duct(Q) = K Q^2，Pa/(m^3/s)^2。"""
    return duct.total_loss_coeff * duct.density / (2.0 * duct.area_m2**2)
