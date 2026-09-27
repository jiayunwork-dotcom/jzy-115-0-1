"""管阻模块测试：流速唯一口径、Q² 律、Q=0 归零、密度影响、非法输入。"""

from __future__ import annotations

import math

import pytest

from app.duct import AIR_DENSITY_20C, Duct
from app.errors import ValidationError


def make_duct(**over) -> Duct:
    kw = {
        "length_m": 20.0,
        "diameter_m": 0.2,
        "friction_factor": 0.02,
        "local_loss_sum": 1.0,
        "density": AIR_DENSITY_20C,
    }
    kw.update(over)
    return Duct(**kw)


def test_default_density_is_20c_air():
    d = Duct(length_m=1.0, diameter_m=0.1, friction_factor=0.02)
    assert d.density == pytest.approx(1.204)


def test_geometry_constants():
    d = make_duct()
    assert d.area_m2 == pytest.approx(math.pi * 0.04 / 4.0)
    # K = fL/D + Σζ = 0.02*20/0.2 + 1 = 3
    assert d.loss_coefficient == pytest.approx(3.0)


def test_velocity_single_formula_consistent_with_pressure():
    """动压/管阻里的流速只能有一个口径：v = Q/A。"""
    d = make_duct()
    q = 0.37
    v_expected = q / (math.pi * 0.2**2 / 4.0)
    assert d.velocity(q) == pytest.approx(v_expected)

    v = d.velocity(q)
    p_dyn_expected = 0.5 * d.density * v**2
    assert d.dynamic_pressure(q) == pytest.approx(p_dyn_expected)
    assert d.required_pressure(q) == pytest.approx(
        d.loss_coefficient * p_dyn_expected
    )


def test_zero_flow_means_zero_resistance():
    """流量为零时管阻压升归零（动压为零）。"""
    d = make_duct()
    assert d.required_pressure(0.0) == 0.0
    assert d.dynamic_pressure(0.0) == 0.0
    assert d.velocity(0.0) == 0.0


def test_pressure_scales_with_flow_squared():
    """核心关系：管阻近似正比于流量平方，流量翻倍压升约四倍。"""
    d = make_duct()
    for q in (0.1, 0.25, 0.4):
        p1 = d.required_pressure(q)
        p2 = d.required_pressure(2.0 * q)
        assert p2 / p1 == pytest.approx(4.0, rel=1e-12)
    # 三倍流量 -> 九倍压升
    assert d.required_pressure(0.3) / d.required_pressure(0.1) == pytest.approx(
        9.0, rel=1e-12
    )


def test_higher_density_means_higher_resistance_same_flow():
    """同一流量下空气密度调高，管阻变大（成正比）。"""
    d1 = make_duct(density=1.0)
    d2 = make_duct(density=1.5)
    q = 0.35
    assert d2.required_pressure(q) > d1.required_pressure(q)
    assert d2.required_pressure(q) / d1.required_pressure(q) == pytest.approx(1.5)


def test_longer_duct_means_more_resistance():
    short = make_duct(length_m=10.0)
    long_ = make_duct(length_m=40.0)
    q = 0.3
    assert long_.required_pressure(q) > short.required_pressure(q)


def test_invalid_inputs_rejected_before_any_computation():
    with pytest.raises(ValidationError, match="管径"):
        make_duct(diameter_m=0.0)
    with pytest.raises(ValidationError, match="管径"):
        make_duct(diameter_m=-0.2)
    with pytest.raises(ValidationError, match="管长"):
        make_duct(length_m=-1.0)
    with pytest.raises(ValidationError, match="密度"):
        make_duct(density=0.0)
    with pytest.raises(ValidationError, match="密度"):
        make_duct(density=-1.2)
    with pytest.raises(ValidationError):
        make_duct(friction_factor=-0.01)
    with pytest.raises(ValidationError):
        make_duct(local_loss_sum=-0.5)
    with pytest.raises(ValidationError):
        make_duct(diameter_m=float("nan"))
    with pytest.raises(ValidationError):
        make_duct(density=float("inf"))


def test_negative_flow_rejected():
    d = make_duct()
    with pytest.raises(ValidationError):
        d.required_pressure(-0.1)
