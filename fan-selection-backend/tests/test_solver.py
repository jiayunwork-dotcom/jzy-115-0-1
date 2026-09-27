"""工作点求根测试：回归基准 + 教师点名的四条物理关系。"""

import pytest

from app.duct import Duct
from app.fan_curve import (
    FanCurveOutOfRange,
    QuadraticFanCurve,
    SampledFanCurve,
)
from app.solver import OperatingPoint, SolverError, solve_operating_point


# ---------- 预置算例：落在两个采样点之间的回归基准 ----------


def test_benchmark_operating_point_matches_closed_form(
    duct: Duct, fan: SampledFanCurve, benchmark_values: dict
) -> None:
    op = solve_operating_point(fan, duct)
    assert 0.2 < op.flow_m3s < 0.3  # 确实落在 0.2、0.3 两个采样点之间
    assert op.flow_m3s == pytest.approx(benchmark_values["flow_m3s"], rel=1e-7)
    assert op.total_pressure_pa == pytest.approx(
        benchmark_values["pressure_pa"], rel=1e-7
    )
    assert op.velocity_ms == pytest.approx(benchmark_values["velocity_ms"], rel=1e-7)
    # 交点必须闭合：风机全压恰好等于风管压升。
    assert op.total_pressure_pa == pytest.approx(op.duct_pressure_pa, abs=1e-6)
    # Q=0 采样点给出风机静压（关断全压）。
    assert op.fan_shutoff_pressure_pa == 80.0


def test_benchmark_hand_check_intermediate_values(
    duct: Duct, fan: SampledFanCurve
) -> None:
    """手算核对：在 Q=0.22045 附近逐段验证，便于学生用计算器复核。"""
    op = solve_operating_point(fan, duct)
    # 0.2~0.3 段风机直线：p = 112 - 240 Q
    fan_p = 112.0 - 240.0 * op.flow_m3s
    assert fan_p == pytest.approx(op.total_pressure_pa, abs=1e-6)
    # 流速与动压的口径检查：v = Q/A, A = pi*0.2^2/4
    assert op.velocity_ms == pytest.approx(
        op.flow_m3s / (3.141592653589793 * 0.2**2 / 4.0), rel=1e-12
    )


# ---------- 关系 1：风管加长，交点流量下降 ----------


def test_longer_duct_shifts_intersection_to_lower_flow(fan: SampledFanCurve) -> None:
    base = Duct(length_m=10.0, diameter_m=0.2, friction_factor=0.02, local_loss_coeff=1.0)
    longer = Duct(length_m=40.0, diameter_m=0.2, friction_factor=0.02, local_loss_coeff=1.0)
    op_short = solve_operating_point(fan, base)
    op_long = solve_operating_point(fan, longer)
    assert op_long.flow_m3s < op_short.flow_m3s
    # 更长的管阻曲线更陡（同一流量压升更大）；新交点沿下降的风机曲线移动，
    # 因此工作点全压应当更高——方向与两条曲线形状一致。
    assert op_long.total_pressure_pa > op_short.total_pressure_pa
    assert op_long.duct_pressure_pa == pytest.approx(
        op_long.total_pressure_pa, abs=1e-6
    )


def test_monotonic_flow_vs_length(fan: SampledFanCurve) -> None:
    flows = []
    for length in [5.0, 10.0, 20.0, 40.0]:
        d = Duct(length_m=length, diameter_m=0.2, friction_factor=0.02, local_loss_coeff=1.0)
        flows.append(solve_operating_point(fan, d).flow_m3s)
    assert flows == sorted(flows, reverse=True)


# ---------- 关系 2：管阻 ~ Q^2（翻倍 -> 四倍，端到端） ----------


def test_doubling_flow_quadruples_duct_pressure(duct: Duct) -> None:
    # 直接的管阻关系在 test_duct 中已盯，这里从工作点侧再确认：
    # 取两个工作点（不同转速），其风管压升比等于流量比的平方。
    from app.affinity import apply_affinity

    fan = SampledFanCurve(
        flows_m3s=(0.0, 0.1, 0.2, 0.3, 0.6, 0.7),
        pressures_pa=(80.0, 72.0, 64.0, 40.0, -200.0, -300.0),
        extrapolation="reject",
    )
    res = apply_affinity(fan, duct, base_speed_rpm=1400.0, new_speed_rpm=2800.0)
    q_ratio = res.new_point.flow_m3s / res.old_point.flow_m3s
    p_ratio = res.new_point.duct_pressure_pa / res.old_point.duct_pressure_pa
    assert p_ratio == pytest.approx(q_ratio**2, rel=1e-6)


# ---------- 关系 3：Q=0 时管阻归零，风机给出静压 ----------


def test_zero_flow_semantics(duct: Duct) -> None:
    from app.duct import resistance_pressure, velocity

    assert resistance_pressure(duct, 0.0) == 0.0
    assert velocity(duct, 0.0) == 0.0
    fan = QuadraticFanCurve(a2=-100.0, a1=-10.0, a0=80.0)
    op = solve_operating_point(fan, duct)
    assert op.fan_shutoff_pressure_pa == 80.0  # 风机静压（动压为 0 时全压=静压）
    assert op.flow_m3s > 0.0


# ---------- 关系 4：密度调高，同一流量管阻变大，交点流量减小 ----------


def test_higher_density_reduces_intersection_flow(fan: SampledFanCurve) -> None:
    base = Duct(10.0, 0.2, 0.02, 1.0, density=1.2)
    dense = Duct(10.0, 0.2, 0.02, 1.0, density=1.5)
    op_base = solve_operating_point(fan, base)
    op_dense = solve_operating_point(fan, dense)
    assert op_dense.flow_m3s < op_base.flow_m3s
    assert op_dense.total_pressure_pa > op_base.total_pressure_pa


# ---------- 越界策略：拒绝 vs 声明式外推 ----------


def test_intersection_beyond_samples_is_rejected(duct: Duct) -> None:
    # 风机在整个采样区间都压得过风管 -> 交点在更大流量侧之外。
    strong = SampledFanCurve(
        flows_m3s=(0.0, 0.1, 0.2, 0.3),
        pressures_pa=(1000.0, 950.0, 900.0, 850.0),
        extrapolation="reject",
    )
    with pytest.raises(FanCurveOutOfRange):
        solve_operating_point(strong, duct)


def test_intersection_below_first_sample_is_rejected() -> None:
    d = Duct(length_m=100.0, diameter_m=0.5, friction_factor=0.05, local_loss_coeff=5.0)
    weak = SampledFanCurve(
        flows_m3s=(0.5, 0.6), pressures_pa=(1.0, 0.0), extrapolation="reject"
    )
    with pytest.raises(FanCurveOutOfRange):
        solve_operating_point(weak, d)


def test_declared_linear_extrapolation_finds_beyond_samples(duct: Duct) -> None:
    f = SampledFanCurve(
        flows_m3s=(0.0, 0.1), pressures_pa=(300.0, 290.0), extrapolation="linear"
    )
    op = solve_operating_point(f, duct)
    assert op.flow_m3s > 0.1  # 交点确实在采样段之外，按事先声明方式外推求得
    assert op.total_pressure_pa == pytest.approx(op.duct_pressure_pa, abs=1e-6)


def test_quadratic_fan_with_negative_shutoff_has_no_operating_point(duct: Duct) -> None:
    fan = QuadraticFanCurve(a2=0.0, a1=0.0, a0=-10.0)
    with pytest.raises(SolverError):
        solve_operating_point(fan, duct)


def test_quadratic_fan_closed_form_case(duct: Duct) -> None:
    """纯二次风机 + 纯二次管阻：构造可解析核对的场景。

    风管 dp = K Q^2（K 由几何决定），风机取 p = a0 - (K + 500) Q^2。
    交点满足 (2K + 500) Q^2 = a0 -> Q = sqrt(a0/(2K+500))。
    """
    import math

    from app.duct import flow_squared_coefficient

    k = flow_squared_coefficient(duct)
    fan = QuadraticFanCurve(a2=-(k + 500.0), a1=0.0, a0=125.0)
    op = solve_operating_point(fan, duct)
    expected_q = math.sqrt(125.0 / (2.0 * k + 500.0))
    assert op.flow_m3s == pytest.approx(expected_q, rel=1e-8)
    assert op.total_pressure_pa == pytest.approx(k * expected_q**2, rel=1e-7)
