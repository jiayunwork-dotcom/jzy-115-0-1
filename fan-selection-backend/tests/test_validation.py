"""输入校验测试：非法输入必须在进入求根之前被挡下。"""

import pytest

from app.duct import Duct
from app.fan_curve import QuadraticFanCurve, SampledFanCurve
from app.validation import (
    ValidationError,
    validate_affinity_case,
    validate_case,
    validate_efficiency,
    validate_speed,
)


def _ok_fan() -> SampledFanCurve:
    return SampledFanCurve((0.0, 0.5), (100.0, 50.0), "reject")


def test_zero_diameter_rejected_before_solving(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.solver as solver_mod

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("非法输入绝不能流入求根过程")

    monkeypatch.setattr(solver_mod, "solve_operating_point", boom)
    d = Duct(length_m=10.0, diameter_m=0.0)
    with pytest.raises(ValidationError, match="管径"):
        validate_case(d, _ok_fan())


def test_negative_length_rejected() -> None:
    with pytest.raises(ValidationError, match="管长"):
        validate_case(Duct(-1.0, 0.2), _ok_fan())


def test_non_positive_density_rejected() -> None:
    with pytest.raises(ValidationError, match="密度"):
        validate_case(Duct(10.0, 0.2, density=0.0), _ok_fan())
    with pytest.raises(ValidationError, match="密度"):
        validate_case(Duct(10.0, 0.2, density=-1.2), _ok_fan())


def test_negative_loss_coefficients_rejected() -> None:
    with pytest.raises(ValidationError, match="摩阻"):
        validate_case(Duct(10.0, 0.2, friction_factor=-0.01), _ok_fan())
    with pytest.raises(ValidationError, match="局部阻力"):
        validate_case(Duct(10.0, 0.2, local_loss_coeff=-2.0), _ok_fan())


def test_empty_fan_curve_rejected() -> None:
    # dataclass 构造层拒绝空曲线；校验层对采样点不足同样给出说明。
    from app.fan_curve import FanCurveError

    with pytest.raises(FanCurveError):
        SampledFanCurve((), ())


def test_speed_and_efficiency_validation() -> None:
    with pytest.raises(ValidationError, match="基准转速"):
        validate_speed(0.0, 1000.0)
    with pytest.raises(ValidationError, match="新转速"):
        validate_speed(1000.0, -5.0)
    with pytest.raises(ValidationError, match="效率"):
        validate_efficiency(1.2)
    validate_efficiency(None)
    validate_efficiency(0.75)


def test_affinity_case_aggregates_validation() -> None:
    with pytest.raises(ValidationError):
        validate_affinity_case(
            Duct(-1.0, 0.0),
            QuadraticFanCurve(0.0, 0.0, 100.0),
            base_speed_rpm=0.0,
            new_speed_rpm=-1.0,
            efficiency=2.0,
        )
