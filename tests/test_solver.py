"""求交点测试：基准回归算例、曲线形状方向、加长降流量、密度降流量、无解与越界。"""

from __future__ import annotations

import math

import pytest
from conftest import bench_analytic_root

from app.duct import Duct
from app.errors import NoIntersectionError, SelectionError
from app.fan import FanCurve, FanPoint
from app.solver import solve_operating_point

pytestmark = pytest.mark.usefixtures("bench_duct", "bench_fan")


def test_benchmark_regression_point_between_samples(bench_duct, bench_fan):
    """固化的手算回归基准：交点必须落在 0.3 与 0.5 两个采样点之间。"""
    op = solve_operating_point(bench_fan, bench_duct)

    # 1) 必须落在给定的两个采样点之间（0.3~0.5 段）。
    assert 0.3 < op.flow_m3s < 0.5

    # 2) 与独立解析解一致（防止求根器自身错误被"自洽"掩盖）。
    q_exact = bench_analytic_root(bench_duct)
    assert op.flow_m3s == pytest.approx(q_exact, abs=1e-7)

    # 3) 交点定义：风机全压 == 风管压升。
    p_duct = bench_duct.required_pressure(op.flow_m3s)
    assert op.total_pressure_pa == pytest.approx(p_duct, abs=1e-6)

    # 4) 回代流速使用同一口径 Q/A。
    assert op.velocity_ms == pytest.approx(
        op.flow_m3s / bench_duct.area_m2, rel=1e-12
    )


def test_benchmark_hand_calc_numbers(bench_duct, bench_fan):
    """可手算核对的数值（K=3, A=π·0.04/4, 段内风机线 415-550Q）。"""
    op = solve_operating_point(bench_fan, bench_duct)
    q = bench_analytic_root(bench_duct)
    # 解析根约 0.3491 m³/s；全压约 223 Pa；流速约 11.1 m/s
    assert op.flow_m3s == pytest.approx(q, abs=1e-7)
    assert 0.34 < op.flow_m3s < 0.36
    assert 215.0 < op.total_pressure_pa < 230.0
    assert 10.5 < op.velocity_ms < 11.7
    # Q=0 时风机静压被保留下来
    assert op.fan_static_pressure_pa == pytest.approx(300.0)


def test_longer_duct_shifts_intersection_flow_down(bench_duct, bench_fan):
    """把风管加长，交点流量应当下降。"""
    q0 = solve_operating_point(bench_fan, bench_duct).flow_m3s
    longer = Duct(
        length_m=60.0,
        diameter_m=bench_duct.diameter_m,
        friction_factor=bench_duct.friction_factor,
        local_loss_sum=bench_duct.local_loss_sum,
        density=bench_duct.density,
    )
    q1 = solve_operating_point(bench_fan, longer).flow_m3s
    assert q1 < q0

    # 方向与曲线形状一致：同一 Q 下新管阻处处更高，风机不变。
    q = q0 * 0.9
    assert longer.required_pressure(q) > bench_duct.required_pressure(q)


def test_intersection_monotonic_in_friction_and_local_loss(bench_duct, bench_fan):
    q0 = solve_operating_point(bench_fan, bench_duct).flow_m3s
    rougher = Duct(
        length_m=bench_duct.length_m,
        diameter_m=bench_duct.diameter_m,
        friction_factor=0.04,
        local_loss_sum=2.0,
        density=bench_duct.density,
    )
    q1 = solve_operating_point(bench_fan, rougher).flow_m3s
    assert q1 < q0


def test_higher_density_reduces_intersection_flow(bench_duct, bench_fan):
    """空气密度调高：同一流量管阻变大，交点流量减小。"""
    q_low = solve_operating_point(bench_fan, bench_duct).flow_m3s
    denser = Duct(
        length_m=bench_duct.length_m,
        diameter_m=bench_duct.diameter_m,
        friction_factor=bench_duct.friction_factor,
        local_loss_sum=bench_duct.local_loss_sum,
        density=bench_duct.density * 2.0,
    )
    q_high = solve_operating_point(bench_fan, denser).flow_m3s
    assert q_high < q_low


def test_zero_flow_residual_balance():
    """Q=0：管阻为 0，风机给静压；风机曲线在该点的取值即静压。"""
    fan = FanCurve.from_samples([(0.0, 300.0), (0.5, 100.0)])
    duct = Duct(10.0, 0.25, 0.02, 1.0, 1.204)
    op = solve_operating_point(fan, duct)
    # 常规情形交点为正流量
    assert op.flow_m3s > 0
    assert duct.required_pressure(0.0) == 0.0
    assert fan.pressure_pa(0.0) == 300.0


def test_quadratic_fan_intersection():
    # 风机 p = 400 - 100 Q²（下降二次），风管 p = c Q²。
    fan = FanCurve.from_quadratic(-100.0, 0.0, 400.0)
    duct = Duct(20.0, 0.2, 0.02, 1.0, 1.204)
    op = solve_operating_point(fan, duct)
    assert op.total_pressure_pa == pytest.approx(
        duct.required_pressure(op.flow_m3s), abs=1e-6
    )
    # 解析：400 = (100 + c) Q²
    c = 0.5 * duct.density * duct.loss_coefficient / duct.area_m2**2
    q_exact = math.sqrt(400.0 / (100.0 + c))
    assert op.flow_m3s == pytest.approx(q_exact, abs=1e-7)


def test_no_intersection_when_fan_always_above_duct():
    # 采样域内风机始终高于管阻（线性下降太慢），应明确报错而不是外推糊弄。
    fan = FanCurve.from_samples(
        [(0.0, 500.0), (0.5, 499.0)], extrapolation="reject"
    )
    duct = Duct(1.0, 0.5, 0.01, 0.0, 1.204)  # 管阻很弱，Q=0.5 仍不到 499 Pa
    with pytest.raises(NoIntersectionError):
        solve_operating_point(fan, duct)


def test_reject_curve_not_covering_zero_raises():
    # 采样不含 Q=0 且声明 reject：求根在 Q=0 评估静压时必须被明确挡下。
    fan = FanCurve.from_samples([(0.1, 300.0), (0.5, 140.0)], "reject")
    duct = Duct(20.0, 0.2, 0.02, 1.0, 1.204)
    with pytest.raises(SelectionError):
        solve_operating_point(fan, duct)


def test_linear_extrapolation_policy_still_solves():
    fan = FanCurve.from_samples(
        [FanPoint(0.1, 320.0), FanPoint(0.5, 100.0)], extrapolation="linear"
    )
    duct = Duct(20.0, 0.2, 0.02, 1.0, 1.204)
    op = solve_operating_point(fan, duct)
    # 交点落在 0.1~0.5 段内
    assert 0.1 < op.flow_m3s < 0.5
    assert op.total_pressure_pa == pytest.approx(
        duct.required_pressure(op.flow_m3s), abs=1e-6
    )
