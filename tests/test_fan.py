"""风机曲线测试：插值、越界拒绝/声明外推、Q=0 静压、二次曲线、空曲线。"""

from __future__ import annotations

import pytest

from app.errors import ExtrapolationError, ValidationError
from app.fan import FanCurve, FanPoint


def test_samples_sorted_and_required_to_be_nonempty():
    with pytest.raises(ValidationError, match="不能为空"):
        FanCurve.from_samples([])
    with pytest.raises(ValidationError):
        FanCurve.from_samples(None)
    with pytest.raises(ValidationError):
        FanCurve.from_samples([FanPoint(0.0, 1.0)])  # 单点无法插值
    with pytest.raises(ValidationError, match="互不相同"):
        FanCurve.from_samples([(0.0, 1.0), (0.0, 2.0)])
def test_unsorted_samples_are_normalized():
    # 乱序但不重复：自动排序后合法，插值结果与正序一致。
    fan = FanCurve.from_samples([(0.5, 140.0), (0.0, 300.0), (0.3, 250.0)])
    assert fan.pressure_pa(0.4) == pytest.approx(195.0)


def test_negative_sample_flow_rejected():
    with pytest.raises(ValidationError):
        FanCurve.from_samples([(-0.1, 300.0), (0.5, 140.0)])


def test_piecewise_linear_interpolation():
    fan = FanCurve.from_samples([(0.0, 300.0), (0.3, 250.0), (0.5, 140.0)])
    assert fan.pressure_pa(0.0) == pytest.approx(300.0)
    assert fan.pressure_pa(0.3) == pytest.approx(250.0)
    assert fan.pressure_pa(0.5) == pytest.approx(140.0)
    # 0.3~0.5 中点
    assert fan.pressure_pa(0.4) == pytest.approx(195.0)
    # 0~0.3 段
    assert fan.pressure_pa(0.15) == pytest.approx(275.0)


def test_shutoff_pressure_at_zero_flow():
    fan = FanCurve.from_samples([(0.0, 300.0), (0.3, 250.0), (0.5, 140.0)])
    # Q=0 时风机给的是静压（关闭压），管阻此时为 0。
    assert fan.static_shutoff_pressure_pa() == pytest.approx(300.0)


def test_out_of_range_rejected_by_default():
    fan = FanCurve.from_samples([(0.1, 300.0), (0.5, 140.0)])  # 不含 0
    with pytest.raises(ExtrapolationError, match="超出"):
        fan.pressure_pa(0.0)
    with pytest.raises(ExtrapolationError):
        fan.pressure_pa(0.6)


def test_declared_linear_extrapolation():
    fan = FanCurve.from_samples(
        [(0.1, 300.0), (0.3, 250.0), (0.5, 140.0)], extrapolation="linear"
    )
    # 右端斜率 (140-250)/(0.5-0.3) = -550，Q=0.6 -> 140-55 = 85
    assert fan.pressure_pa(0.6) == pytest.approx(85.0)
    # 左端斜率 (250-300)/(0.3-0.1) = -250，Q=0 -> 300 - (-250)*(0.1) = 325
    assert fan.pressure_pa(0.0) == pytest.approx(325.0)


def test_bad_extrapolation_mode():
    with pytest.raises(ValidationError):
        FanCurve.from_samples([(0.0, 1.0), (1.0, 2.0)], extrapolation="silent_guess")  # type: ignore[arg-type]


def test_quadratic_curve():
    fan = FanCurve.from_quadratic(a2=-200.0, a1=-100.0, a0=300.0)
    assert fan.pressure_pa(0.0) == pytest.approx(300.0)
    assert fan.pressure_pa(0.5) == pytest.approx(300.0 - 50.0 - 50.0)
    # 多项式曲线没有采样域边界
    assert fan.q_max == float("inf")


def test_quadratic_requires_finite_coefficients():
    with pytest.raises(ValidationError):
        FanCurve.from_quadratic(float("nan"), 0.0, 1.0)
    with pytest.raises(ValidationError):
        FanCurve.from_quadratic(1.0, float("inf"), 1.0)


def test_shaft_power_interpolation():
    fan = FanCurve.from_samples(
        [FanPoint(0.0, 300.0, 120.0), FanPoint(0.5, 140.0, 90.0)]
    )
    assert fan.shaft_power_w(0.25) == pytest.approx(105.0)
    fan2 = FanCurve.from_samples([(0.0, 300.0), (0.5, 140.0)])
    assert fan2.shaft_power_w(0.25) is None
