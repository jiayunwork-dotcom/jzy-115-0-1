"""相似律换算测试。

盯住两组关系：
  Q2/Q1 = n2/n1，p2/p1 = (n2/n1)^2，P2/P1 = (n2/n1)^3；
以及一条纪律：
  新交点必须由变换后的整条风机曲线重新求根得到，
  不允许拿旧交点流量直接乘转速比。
"""

import pytest

from app import affinity as affinity_module
from app.affinity import apply_affinity, scale_fan_curve
from app.duct import Duct
from app.fan_curve import (
    QuadraticFanCurve,
    SampledFanCurve,
)


def test_affinity_ratios_for_speed_doubling(
    duct: Duct, fan: SampledFanCurve
) -> None:
    # 基准风机采样域只到 0.3，加倍后交点 ~0.44，需要给变换后的曲线
    # 留出采样点——这里改用一条覆盖更宽的 reject 曲线。
    fan_wide = SampledFanCurve(
        flows_m3s=(0.0, 0.1, 0.2, 0.3, 0.6, 0.7),
        pressures_pa=(80.0, 72.0, 64.0, 40.0, -200.0, -300.0),
        extrapolation="reject",
    )
    res = apply_affinity(fan_wide, duct, 1400.0, 2800.0)
    assert res.speed_ratio == pytest.approx(2.0)
    assert res.flow_ratio == pytest.approx(2.0, abs=1e-7)
    assert res.pressure_ratio == pytest.approx(4.0, abs=1e-6)
    assert res.air_power_ratio == pytest.approx(8.0, abs=1e-6)
    # 两个工作点都必须独立闭合。
    assert res.old_point.total_pressure_pa == pytest.approx(
        res.old_point.duct_pressure_pa, abs=1e-6
    )
    assert res.new_point.total_pressure_pa == pytest.approx(
        res.new_point.duct_pressure_pa, abs=1e-6
    )


def test_affinity_ratios_general_speeds(duct: Duct) -> None:
    fan = QuadraticFanCurve(a2=-400.0, a1=-20.0, a0=120.0)
    for n1, n2 in [(1200.0, 1500.0), (1000.0, 800.0), (900.0, 1800.0)]:
        r = n2 / n1
        res = apply_affinity(fan, duct, n1, n2)
        assert res.flow_ratio == pytest.approx(r, abs=1e-7)
        assert res.pressure_ratio == pytest.approx(r**2, abs=1e-6)
        assert res.air_power_ratio == pytest.approx(r**3, abs=1e-6)


def test_shaft_power_cube_law_with_efficiency(duct: Duct) -> None:
    fan = QuadraticFanCurve(a2=-400.0, a1=-20.0, a0=120.0)
    res = apply_affinity(fan, duct, 1000.0, 1500.0, fan_total_efficiency=0.6)
    r = 1.5
    assert res.shaft_power_ratio == pytest.approx(r**3, abs=1e-6)
    assert res.shaft_power_old_w == pytest.approx(
        res.air_power_old_w / 0.6, rel=1e-12
    )
    assert res.shaft_power_new_w == pytest.approx(
        res.air_power_new_w / 0.6, rel=1e-12
    )


def test_new_intersection_is_re_solved_not_scaled(
    duct: Duct, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实现纪律：apply_affinity 必须对新曲线再次调用求根器。

    如果有人偷偷改成"旧交点流量 * 转速比"直接交差，这个测试
    （统计 solve_operating_point 调用次数）会挂掉。
    """
    fan = QuadraticFanCurve(a2=-400.0, a1=-20.0, a0=120.0)
    calls: list[tuple[float, float]] = []
    original = affinity_module.solve_operating_point

    def spy(f, d, **kwargs):  # type: ignore[no-untyped-def]
        op = original(f, d, **kwargs)
        calls.append((op.flow_m3s, op.total_pressure_pa))
        return op

    monkeypatch.setattr(affinity_module, "solve_operating_point", spy)
    res = apply_affinity(fan, duct, 1000.0, 1500.0)

    assert res.re_solved is True
    assert len(calls) == 2  # 旧曲线一次、变换后的新曲线一次
    # 第二次求根喂进去的风机曲线必须是变换后的，而不是原曲线。
    assert calls[1][0] != calls[0][0]


def test_scaled_curve_moves_every_sample_point(duct: Duct) -> None:
    """曲线整体变换：逐点 (Q,p) -> (r Q, r^2 p)。"""
    fan = SampledFanCurve(
        flows_m3s=(0.0, 0.2, 0.4),
        pressures_pa=(100.0, 60.0, 0.0),
        extrapolation="reject",
    )
    scaled = scale_fan_curve(fan, 1.5)
    assert isinstance(scaled, SampledFanCurve)
    assert scaled.flows_m3s == pytest.approx((0.0, 0.3, 0.6))
    assert scaled.pressures_pa == pytest.approx((225.0, 135.0, 0.0))


def test_scaled_quadratic_coefficients(duct: Duct) -> None:
    """二次曲线整体变换：a2 不变，a1 -> r a1，a0 -> r^2 a0。"""
    fan = QuadraticFanCurve(a2=-400.0, a1=-20.0, a0=120.0)
    scaled = scale_fan_curve(fan, 1.5)
    assert isinstance(scaled, QuadraticFanCurve)
    assert scaled.a2 == pytest.approx(-400.0)
    assert scaled.a1 == pytest.approx(-30.0)
    assert scaled.a0 == pytest.approx(270.0)
    # 与逐点变换等价：旧点 (Q,p) 在新曲线上的对应点是 (1.5Q, 2.25p)。
    q = 0.2
    assert scaled.evaluate(1.5 * q) == pytest.approx(2.25 * fan.evaluate(q), rel=1e-12)


def test_reduced_speed_lowers_flow_and_pressure(duct: Duct) -> None:
    fan = QuadraticFanCurve(a2=-400.0, a1=-20.0, a0=120.0)
    res = apply_affinity(fan, duct, 1500.0, 1000.0)
    assert res.new_point.flow_m3s < res.old_point.flow_m3s
    assert res.new_point.total_pressure_pa < res.old_point.total_pressure_pa
