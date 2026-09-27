"""风机曲线求值模块测试：插值、显式拒绝、声明式外推。"""

import pytest

from app.fan_curve import (
    FanCurveError,
    FanCurveOutOfRange,
    QuadraticFanCurve,
    SampledFanCurve,
)


def test_sampled_interpolation_and_exact_samples(fan: SampledFanCurve) -> None:
    assert fan.evaluate(0.0) == 80.0
    assert fan.evaluate(0.1) == 72.0
    assert fan.evaluate(0.2) == 64.0
    assert fan.evaluate(0.3) == 40.0
    # 线性插值
    assert fan.evaluate(0.15) == pytest.approx(68.0)
    assert fan.evaluate(0.25) == pytest.approx(52.0)


def test_reject_policy_refuses_outside_range(fan: SampledFanCurve) -> None:
    """落在采样范围之外时必须明确拒绝，不能默默给数。"""
    with pytest.raises(FanCurveOutOfRange):
        fan.evaluate(0.31)
    other = SampledFanCurve(
        flows_m3s=(0.1, 0.2, 0.3),
        pressures_pa=(72.0, 64.0, 40.0),
        extrapolation="reject",
    )
    with pytest.raises(FanCurveOutOfRange):
        other.evaluate(0.05)


def test_linear_extrapolation_is_declared_and_predictable() -> None:
    """声明 linear 后按边界段线性外推，斜率可预测。"""
    f = SampledFanCurve(
        flows_m3s=(0.0, 0.5),
        pressures_pa=(100.0, 90.0),
        extrapolation="linear",
    )
    # 末段斜率 -20 Pa/(m^3/s)
    assert f.evaluate(0.75) == pytest.approx(85.0)
    # 首段向更小流量（Q<0 仍被物理拒绝）
    with pytest.raises(FanCurveOutOfRange):
        f.evaluate(-0.1)


def test_empty_or_single_sample_is_rejected() -> None:
    with pytest.raises(FanCurveError):
        SampledFanCurve(flows_m3s=(), pressures_pa=())
    with pytest.raises(FanCurveError):
        SampledFanCurve(flows_m3s=(0.2,), pressures_pa=(50.0,))


def test_samples_must_be_strictly_increasing() -> None:
    with pytest.raises(FanCurveError):
        SampledFanCurve(
            flows_m3s=(0.1, 0.2, 0.2),
            pressures_pa=(70.0, 60.0, 50.0),
        )


def test_quadratic_curve() -> None:
    f = QuadraticFanCurve(a2=-100.0, a1=-10.0, a0=80.0)
    assert f.evaluate(0.0) == 80.0
    assert f.evaluate(0.2) == pytest.approx(-4.0 - 2.0 + 80.0)
    with pytest.raises(FanCurveOutOfRange):
        f.evaluate(-1.0)
