"""FastAPI 应用入口。

领域错误（非法输入、越界外推、无交点等）统一转成 HTTP 422 与中文说明，
保证任何非法输入都不会流进求根过程。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .errors import SelectionError
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="风机-风管工作点选型核算服务",
        version="1.0.0",
        description=(
            "单管 + 单台风机：括号法求解风机特性曲线与风管阻力曲线的稳定交点，"
            "并按相似律在不同转速间换算（新转速下重新求交）。"
        ),
    )

    @app.exception_handler(SelectionError)
    async def _selection_error_handler(_: Request, exc: SelectionError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "RequestValidationError",
                "detail": "请求参数未通过校验",
                "issues": jsonable_encoder(exc.errors()),
            },
        )

    app.include_router(router)
    return app


app = create_app()
