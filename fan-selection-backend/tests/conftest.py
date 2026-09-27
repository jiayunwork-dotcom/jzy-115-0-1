"""pytest 公共夹具：预置的手算算例（也是 HTTP 回归基准）。

算例：
  风管 D=0.2 m, L=10 m, lambda=0.02, sum(zeta)=1.0, rho=1.2 kg/m^3
    A = pi D^2/4
    K_zeta = lambda L/D + sum(zeta) = 1.0 + 1.0 = 2.0
    dp(Q) = K_zeta * rho/2 * (Q/A)^2
    系数 K = K_zeta * rho/(2 A^2)
  风机采样点 (0,80),(0.1,72),(0.2,64),(0.3,40)，策略 reject。
    0.2~0.3 段为直线 p(Q) = 112 - 240 Q。
    交点方程 K Q^2 + 240 Q - 112 = 0，
    解析根 Q* = (-240 + sqrt(240^2 + 4*K*112))/(2K) ≈ 0.2204547 m^3/s，
    位于 0.2 与 0.3 两个采样点之间，p* ≈ 59.0909 Pa，v* ≈ 7.0173 m/s。
"""

import math

import pytest

from app.duct import Duct, flow_squared_coefficient
from app.fan_curve import SampledFanCurve


@pytest.fixture
def duct() -> Duct:
    return Duct(
        length_m=10.0,
        diameter_m=0.2,
        friction_factor=0.02,
        local_loss_coeff=1.0,
        density=1.2,
    )


@pytest.fixture
def fan() -> SampledFanCurve:
    return SampledFanCurve(
        flows_m3s=(0.0, 0.1, 0.2, 0.3),
        pressures_pa=(80.0, 72.0, 64.0, 40.0),
        extrapolation="reject",
    )


@pytest.fixture
def benchmark_values(duct) -> dict:
    """独立于求根器、用求根公式算出来的基准值。"""
    k = flow_squared_coefficient(duct)
    q = (-240.0 + math.sqrt(240.0**2 + 4.0 * k * 112.0)) / (2.0 * k)
    p = k * q**2
    area = math.pi * duct.diameter_m**2 / 4.0
    return {
        "flow_m3s": q,
        "pressure_pa": p,
        "velocity_ms": q / area,
        "k": k,
        "shutoff_pressure_pa": 80.0,
    }


@pytest.fixture
def example_payload() -> dict:
    """与 GET /example 完全一致的请求体，供 HTTP 回归测试使用。"""
    return {
        "duct": {
            "length_m": 10.0,
            "diameter_m": 0.2,
            "friction_factor": 0.02,
            "local_loss_coeff": 1.0,
            "density": 1.2,
        },
        "fan": {
            "kind": "sampled",
            "flows_m3s": [0.0, 0.1, 0.2, 0.3],
            "pressures_pa": [80.0, 72.0, 64.0, 40.0],
            "extrapolation": "reject",
            "max_flow_m3s": 100.0,
        },
    }
