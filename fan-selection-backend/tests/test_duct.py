"""风管阻力模块测试。

重点盯住：
* 流速只有 Q/A 这一个口径，管阻内部与返回值一致；
* 管阻严格正比于流量平方（流量翻倍、压升约四倍）；
* Q=0 时管阻归零；
* 空气密度调高，同一流量下管阻线性变大。
"""

import pytest

from app.duct import (
    AIR_DENSITY_20C,
    Duct,
    dynamic_pressure,
    flow_squared_coefficient,
    resistance_pressure,
    velocity,
)


def test_default_density_is_20c_air() -> None:
    d = Duct(length_m=1.0, diameter_m=0.5)
    assert d.density == AIR_DENSITY_20C == 1.2


def test_velocity_single_definition(duct: Duct) -> None:
    import math

    q = 0.22
    expected_v = q / (math.pi * 0.2**2 / 4.0)
    v = velocity(duct, q)
    assert v == pytest.approx(expected_v, rel=1e-14)
    # 动压必须用同一个 v，管阻必须复用同一条计算路径。
    pd = dynamic_pressure(duct, q)
    assert pd == pytest.approx(0.5 * duct.density * v**2, rel=1e-14)
    assert resistance_pressure(duct, q) == pytest.approx(
        duct.total_loss_coeff * pd, rel=1e-14
    )


def test_resistance_is_proportional_to_flow_squared(duct: Duct) -> None:
    """流量翻倍，管阻压升约四倍（教师明确要盯的关系）。"""
    p1 = resistance_pressure(duct, 0.10)
    p2 = resistance_pressure(duct, 0.20)
    assert p2 / p1 == pytest.approx(4.0, rel=1e-12)

    p3 = resistance_pressure(duct, 0.30)
    assert p3 / p1 == pytest.approx(9.0, rel=1e-12)

    # 用 K Q^2 形式交叉验证。
    k = flow_squared_coefficient(duct)
    for q in [0.05, 0.123, 0.27]:
        assert resistance_pressure(duct, q) == pytest.approx(k * q**2, rel=1e-12)


def test_resistance_zero_at_zero_flow(duct: Duct) -> None:
    """Q=0：流速为 0、动压为 0、管阻压升归零。"""
    assert velocity(duct, 0.0) == 0.0
    assert dynamic_pressure(duct, 0.0) == 0.0
    assert resistance_pressure(duct, 0.0) == 0.0


def test_higher_density_gives_higher_resistance(duct: Duct) -> None:
    """同一流量、同一截面：密度调高，管阻线性变大。"""
    denser = Duct(
        length_m=duct.length_m,
        diameter_m=duct.diameter_m,
        friction_factor=duct.friction_factor,
        local_loss_coeff=duct.local_loss_coeff,
        density=duct.density * 1.2,
    )
    q = 0.2
    assert resistance_pressure(denser, q) == pytest.approx(
        1.2 * resistance_pressure(duct, q), rel=1e-13
    )
    # 流速只取决于几何与流量，密度不影响流速口径。
    assert velocity(denser, q) == pytest.approx(velocity(duct, q), rel=1e-14)


def test_longer_pipe_only_changes_friction_term() -> None:
    short = Duct(length_m=5.0, diameter_m=0.3, friction_factor=0.03, local_loss_coeff=2.0)
    long = Duct(length_m=20.0, diameter_m=0.3, friction_factor=0.03, local_loss_coeff=2.0)
    q = 0.4
    ratio = resistance_pressure(long, q) / resistance_pressure(short, q)
    # K_zeta: 0.03*5/0.3+2 = 2.5  vs  0.03*20/0.3+2 = 4.0
    assert ratio == pytest.approx(4.0 / 2.5, rel=1e-13)
