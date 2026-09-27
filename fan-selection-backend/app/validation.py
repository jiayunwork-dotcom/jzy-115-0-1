"""输入校验模块：在进入求根过程之前挡掉一切非法输入。

校验针对的是已经构造好的领域对象（Duct / FanCurve / 转速等），
与 HTTP 层的 Pydantic schema 解耦：核心计算从不依赖 Web 框架。
所有错误聚合成一条带中文说明的 :class:`ValidationError`。
"""

from __future__ import annotations

from .duct import Duct
from .fan_curve import FanCurve, QuadraticFanCurve, SampledFanCurve


class ValidationError(ValueError):
    """请求在物理/数值上不合法，附中文说明。"""


def validate_duct(duct: Duct) -> None:
    problems: list[str] = []
    if duct.diameter_m <= 0:
        problems.append(f"管径必须为正（收到 {duct.diameter_m:g} m），管径为零无法定义截面")
    if duct.length_m < 0:
        problems.append(f"管长不能为负（收到 {duct.length_m:g} m）")
    if duct.friction_factor < 0:
        problems.append(f"沿程摩阻系数不能为负（收到 {duct.friction_factor:g}）")
    if duct.local_loss_coeff < 0:
        problems.append(f"局部阻力系数之和不能为负（收到 {duct.local_loss_coeff:g}）")
    if duct.density <= 0:
        problems.append(f"空气密度必须为正（收到 {duct.density:g} kg/m^3）")
    if problems:
        raise ValidationError("；".join(problems))


def validate_fan_curve(fan: FanCurve) -> None:
    problems: list[str] = []
    if isinstance(fan, SampledFanCurve):
        if len(fan.flows_m3s) == 0:
            problems.append("风机曲线为空：至少需要 2 组流量-全压采样点")
        elif len(fan.flows_m3s) < 2:
            problems.append("采样点过少：至少需要 2 组流量-全压采样点")
        elif any(
            cur <= prev
            for prev, cur in zip(fan.flows_m3s, fan.flows_m3s[1:])
        ):
            problems.append("采样点流量必须严格递增且不得重复")
        if fan.max_flow_m3s <= 0:
            problems.append("外推流量上限必须为正")
    elif isinstance(fan, QuadraticFanCurve):
        # 系数本身允许为任意实数（风机曲线通常 a2<0），无需限制符号。
        if fan.max_flow_m3s <= 0:
            problems.append("曲线流量上限必须为正")
    else:  # pragma: no cover - 类型穷举保护
        problems.append(f"未知的风机曲线类型: {type(fan).__name__}")
    if problems:
        raise ValidationError("；".join(problems))


def validate_speed(base_speed_rpm: float, new_speed_rpm: float | None = None) -> None:
    problems: list[str] = []
    if base_speed_rpm <= 0:
        problems.append(f"基准转速必须为正（收到 {base_speed_rpm:g} rpm）")
    if new_speed_rpm is not None and new_speed_rpm <= 0:
        problems.append(f"新转速必须为正（收到 {new_speed_rpm:g} rpm）")
    if problems:
        raise ValidationError("；".join(problems))


def validate_efficiency(efficiency: float | None) -> None:
    if efficiency is None:
        return
    if not (0.0 < efficiency <= 1.0):
        raise ValidationError(
            f"风机全压效率必须落在 (0, 1] 内（收到 {efficiency:g}）"
        )


def validate_case(duct: Duct, fan: FanCurve) -> None:
    """工作点求解前的完整校验。"""
    validate_duct(duct)
    validate_fan_curve(fan)


def validate_affinity_case(
    duct: Duct,
    fan: FanCurve,
    base_speed_rpm: float,
    new_speed_rpm: float,
    efficiency: float | None,
) -> None:
    """相似律换算前的完整校验（任何一项不过都不进入求根）。"""
    validate_duct(duct)
    validate_fan_curve(fan)
    validate_speed(base_speed_rpm, new_speed_rpm)
    validate_efficiency(efficiency)
