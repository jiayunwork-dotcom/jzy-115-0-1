"""相似律测试：Q∝n、p∝n²、P∝n³；且新交点是重新求解而不是直接乘比例。"""

from __future__ import annotations

import pytest

from app.affinity import scale_fan_curve, solve_at_speed
from app.errors import ValidationError
from app.solver import solve_operating_point


def test_speed_up_increases_operating_values(bench_duct, bench_fan):
    sol = solve_at_speed(bench_fan, bench_duct, base_speed_rpm=1000.0, new_speed_rpm=1200.0)
    r = 1.2
    assert sol.speed_ratio == pytest.approx(r)
    assert sol.new.flow_m3s > sol.base.flow_m3s
    assert sol.new.total_pressure_pa > sol.base.total_pressure_pa


def test_affinity_flow_linear_in_speed(bench_duct, bench_fan):
    """流量与转速成正比。二次齐次管阻下新交点的比值应精确等于转速比。"""
    for r in (0.8, 1.0, 1.25):
        sol = solve_at_speed(bench_fan, bench_duct, 1000.0, 1000.0 * r)
        assert sol.flow_ratio == pytest.approx(r, abs=1e-6)
        assert sol.new.flow_m3s == pytest.approx(sol.base.flow_m3s * r, abs=1e-7)


def test_affinity_pressure_quadratic_in_speed(bench_duct, bench_fan):
    """全压与转速平方成正比（在重新求出的新交点处核对）。"""
    for r in (0.8, 1.0, 1.25):
        sol = solve_at_speed(bench_fan, bench_duct, 900.0, 900.0 * r)
        assert sol.pressure_ratio == pytest.approx(r**2, abs=1e-5)
        assert sol.new.total_pressure_pa == pytest.approx(
            sol.base.total_pressure_pa * r**2, abs=1e-6
        )


def test_affinity_power_cubic_in_speed(bench_duct, bench_fan):
    """轴功率与转速立方成正比（基准算例带轴功率采样）。"""
    r = 1.2
    sol = solve_at_speed(bench_fan, bench_duct, 1000.0, 1200.0)
    assert sol.power_ratio is not None
    assert sol.power_ratio == pytest.approx(r**3, rel=1e-4)
    assert sol.new.shaft_power_w == pytest.approx(
        sol.base.shaft_power_w * r**3, rel=1e-4
    )


def test_scaled_curve_pointwise_affinity_laws(bench_fan):
    """整体曲线变换后逐点满足 (rQ, r²p, r³P)，证明不是只搬旧交点。"""
    r = 1.5
    scaled = scale_fan_curve(bench_fan, r)
    for old, new in zip(bench_fan.points, scaled.points):
        assert new.flow_m3s == pytest.approx(old.flow_m3s * r)
        assert new.total_pressure_pa == pytest.approx(old.total_pressure_pa * r**2)
        assert new.shaft_power_w == pytest.approx(old.shaft_power_w * r**3)


def test_new_intersection_is_recomputed_against_duct(bench_duct, bench_fan):
    """新交点必须真的落在新曲线上且满足管阻平衡，而不是旧 Q 乘比例。"""
    r = 1.3
    sol = solve_at_speed(bench_fan, bench_duct, 1000.0, 1300.0)
    scaled = scale_fan_curve(bench_fan, r)

    # 新点对新风机曲线闭合
    assert scaled.pressure_pa(sol.new.flow_m3s) == pytest.approx(
        sol.new.total_pressure_pa, abs=1e-6
    )
    # 新点对原风管闭合
    assert bench_duct.required_pressure(sol.new.flow_m3s) == pytest.approx(
        sol.new.total_pressure_pa, abs=1e-6
    )

    # 旧点放到新转速下不再是工作点（旧 Q 旧p 不满足新曲线/管阻平衡）。
    base_q = sol.base.flow_m3s
    residual_at_old_q = (
        scaled.pressure_pa(base_q) - bench_duct.required_pressure(base_q)
    )
    assert abs(residual_at_old_q) > 1.0  # Pa，旧交点位置明显失衡


def test_speed_down_below_base(bench_duct, bench_fan):
    r = 0.6
    sol = solve_at_speed(bench_fan, bench_duct, 1500.0, 900.0)
    assert sol.speed_ratio == pytest.approx(r)
    assert sol.new.flow_m3s == pytest.approx(sol.base.flow_m3s * r, abs=1e-7)
    assert sol.new.total_pressure_pa == pytest.approx(
        sol.base.total_pressure_pa * r**2, abs=1e-6
    )


def test_quadratic_fan_scaling_coefficients():
    from app.fan import FanCurve

    fan = FanCurve.from_quadratic(-100.0, -50.0, 400.0)
    r = 2.0
    scaled = scale_fan_curve(fan, r)
    # p'(Q)=a2 Q² + a1·r Q + a0·r²
    assert scaled.a2 == pytest.approx(-100.0)
    assert scaled.a1 == pytest.approx(-100.0)
    assert scaled.a0 == pytest.approx(1600.0)
    # 等价地，整条曲线逐点满足相似律
    for q in (0.0, 0.3, 0.5):
        assert scaled.pressure_pa(r * q) == pytest.approx(r**2 * fan.pressure_pa(q))


def test_invalid_speeds_rejected(bench_duct, bench_fan):
    with pytest.raises(ValidationError):
        solve_at_speed(bench_fan, bench_duct, 0.0, 1000.0)
    with pytest.raises(ValidationError):
        solve_at_speed(bench_fan, bench_duct, 1000.0, -5.0)
    with pytest.raises(ValidationError):
        scale_fan_curve(bench_fan, 0.0)


def test_base_point_matches_direct_solve(bench_duct, bench_fan):
    sol = solve_at_speed(bench_fan, bench_duct, 1000.0, 1100.0)
    direct = solve_operating_point(bench_fan, bench_duct)
    assert sol.base.flow_m3s == pytest.approx(direct.flow_m3s, abs=1e-9)
