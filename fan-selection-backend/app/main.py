"""HTTP 层：FastAPI 应用与路由。

只做三件事：请求 schema -> 领域对象的映射、调用校验与求解/相似律、
把领域结果翻译成响应 schema。核心计算不依赖本模块。

路由
----
POST /operating-point   给定风管 + 风机曲线，求当前稳定工作点
POST /affinity          在其基础上指定新转速，曲线整体变换后重新求交
GET  /example           返回可手算核对的预置算例请求体（回归基准）
GET  /health            存活探针
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .affinity import AffinityResult, apply_affinity
from .duct import Duct
from .fan_curve import (
    FanCurve,
    FanCurveError,
    QuadraticFanCurve,
    SampledFanCurve,
)
from .schemas import (
    AffinityOutput,
    AffinityRequest,
    OperatingPointOutput,
    OperatingPointRequest,
    PowerOutput,
)
from .solver import OperatingPoint, SolverError, solve_operating_point
from .validation import (
    ValidationError,
    validate_affinity_case,
    validate_case,
)

app = FastAPI(
    title="风机选型核算后端",
    version=__version__,
    description=(
        "单管 + 单台风机的稳定工作点求解与转速相似律换算。\n\n"
        "范围限定：不做多环管网平差，不提供前端页面。"
    ),
)


# ----------------------- schema -> 领域对象映射 -----------------------


def _to_duct(inp: OperatingPointRequest | AffinityRequest) -> Duct:
    d = inp.duct
    return Duct(
        length_m=d.length_m,
        diameter_m=d.diameter_m,
        friction_factor=d.friction_factor,
        local_loss_coeff=d.local_loss_coeff,
        density=d.density,
    )


def _to_fan(inp: OperatingPointRequest | AffinityRequest) -> FanCurve:
    f = inp.fan
    if f.kind == "sampled":
        return SampledFanCurve(
            flows_m3s=tuple(f.flows_m3s),
            pressures_pa=tuple(f.pressures_pa),
            extrapolation=f.extrapolation,
            max_flow_m3s=f.max_flow_m3s,
        )
    return QuadraticFanCurve(
        a2=f.a2, a1=f.a1, a0=f.a0, max_flow_m3s=f.max_flow_m3s
    )


def _point_output(op: OperatingPoint) -> OperatingPointOutput:
    return OperatingPointOutput(
        flow_m3s=op.flow_m3s,
        total_pressure_pa=op.total_pressure_pa,
        duct_pressure_pa=op.duct_pressure_pa,
        velocity_ms=op.velocity_ms,
        iterations=op.iterations,
        fan_shutoff_pressure_pa=op.fan_shutoff_pressure_pa,
    )


def _affinity_output(res: AffinityResult) -> AffinityOutput:
    return AffinityOutput(
        speed_ratio=res.speed_ratio,
        re_solved=res.re_solved,
        old=_point_output(res.old_point),
        new=_point_output(res.new_point),
        flow_ratio=res.flow_ratio,
        pressure_ratio=res.pressure_ratio,
        air_power=PowerOutput(
            old_w=res.air_power_old_w,
            new_w=res.air_power_new_w,
            ratio=res.air_power_ratio,
        ),
        shaft_power=(
            PowerOutput(
                old_w=res.shaft_power_old_w,
                new_w=res.shaft_power_new_w,
                ratio=res.shaft_power_ratio,
            )
            if res.shaft_power_ratio is not None
            else None
        ),
    )


# ----------------------- 统一错误响应 -----------------------


def _error_response(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": code, "detail": detail}
    )


@app.exception_handler(ValidationError)
async def _validation_error_handler(_: Request, exc: ValidationError) -> JSONResponse:
    # 所有在求解前拦下的物理/数值非法输入统一 422。
    return _error_response(422, "invalid_input", str(exc))


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error_response(
        422,
        "invalid_request",
        f"请求体不符合接口定义：{exc.errors()}",
    )


@app.exception_handler(FanCurveError)
async def _fan_curve_error_handler(_: Request, exc: FanCurveError) -> JSONResponse:
    # 含 FanCurveOutOfRange：超出声明范围且策略为拒绝。
    return _error_response(422, "fan_curve_error", str(exc))


@app.exception_handler(SolverError)
async def _solver_error_handler(_: Request, exc: SolverError) -> JSONResponse:
    # 找不到稳定工作点属于"算不出来"而不是请求格式错误，用 409 表明冲突。
    return _error_response(409, "no_operating_point", str(exc))


# ----------------------- 路由 -----------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/operating-point", response_model=OperatingPointOutput)
def operating_point(req: OperatingPointRequest) -> OperatingPointOutput:
    """给定风管与风机曲线，求当前转速下的稳定工作点。"""
    duct = _to_duct(req)
    fan = _to_fan(req)
    validate_case(duct, fan)  # 非法输入在进入求根前就被挡下
    return _point_output(solve_operating_point(fan, duct))


@app.post("/affinity", response_model=AffinityOutput)
def affinity(req: AffinityRequest) -> AffinityOutput:
    """指定新转速：曲线按相似律整体变换后，重新求解与同一风管的交点。"""
    duct = _to_duct(req)
    fan = _to_fan(req)
    validate_affinity_case(
        duct,
        fan,
        req.base_speed_rpm,
        req.new_speed_rpm,
        req.fan_total_efficiency,
    )
    result = apply_affinity(
        fan,
        duct,
        base_speed_rpm=req.base_speed_rpm,
        new_speed_rpm=req.new_speed_rpm,
        fan_total_efficiency=req.fan_total_efficiency,
    )
    return _affinity_output(result)


# ----------------------- 预置手算算例（回归基准） -----------------------

EXAMPLE_REQUEST: dict = {
    "description": (
        "直管 + 内联风机手算算例：D=0.2 m，L=10 m，lambda=0.02，"
        "sum(zeta)=1.0，rho=1.2 kg/m^3；"
        "风机四个采样点，交点落在 0.2 与 0.3 m^3/s 两个采样点之间，"
        "零流量采样点给出风机关断静压 80 Pa。"
    ),
    "duct": {
        "length_m": 10.0,
        "diameter_m": 0.2,
        "friction_factor": 0.02,
        "local_loss_coeff": 1.0,
        "density": 1.2,
    },
    "fan": {
        "kind": "sampled",
        "flows_m3s": [0.0, 0.1, 0.2, 0.3],
        "pressures_pa": [80.0, 72.0, 64.0, 40.0],
        "extrapolation": "reject",
        "max_flow_m3s": 100.0,
    },
}


@app.get("/example")
def example() -> dict:
    """返回预置算例的请求体，可直接 POST 给 /operating-point。"""
    return EXAMPLE_REQUEST
