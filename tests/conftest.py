import math
import os
import sys

import pytest

# 让 tests 能直接 import app（即使没安装为包）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.duct import Duct
from app.fan import FanCurve, FanPoint


@pytest.fixture
def bench_duct() -> Duct:
    """README 中可手算核对的基准风管：L=20m, D=0.2m, f=0.02, Σζ=1, ρ=1.204。

    K = fL/D + Σζ = 0.02*20/0.2 + 1 = 3
    """
    return Duct(
        length_m=20.0,
        diameter_m=0.2,
        friction_factor=0.02,
        local_loss_sum=1.0,
        density=1.204,
    )


@pytest.fixture
def bench_fan() -> FanCurve:
    """基准内联风机：线性分段采样（含轴功率采样），默认 reject 越界。"""
    return FanCurve.from_samples(
        [
            FanPoint(0.0, 300.0, 120.0),
            FanPoint(0.3, 250.0, 110.0),
            FanPoint(0.5, 140.0, 90.0),
        ],
        extrapolation="reject",
    )


def bench_analytic_root(duct: Duct) -> float:
    """独立的基准解析解：交点位于 0.3~0.5 段，该段风机线 415-550Q。

    dP = (0.5·ρ·K/A²)·Q²，解 415 - 550Q - cQ² = 0 的正根。
    """
    k = duct.loss_coefficient
    c = 0.5 * duct.density * k / duct.area_m2**2
    disc = 550.0**2 + 4.0 * c * 415.0
    return (-550.0 + math.sqrt(disc)) / (2.0 * c)
