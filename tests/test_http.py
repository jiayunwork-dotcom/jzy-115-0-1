"""HTTP 端到端测试（FastAPI TestClient）。"""

from __future__ import annotations

import pytest
from conftest import bench_analytic_root  # type: ignore
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def bench_payload(**fan_over) -> dict:
    duct = {
        "length_m": 20.0,
        "diameter_m": 0.2,
        "friction_factor": 0.02,
        "local_loss_sum": 1.0,
        "density": 1.204,
    }
    fan = {
        "samples": [
            {"flow_m3s": 0.0, "total_pressure_pa": 300.0, "shaft_power_w": 120.0},
            {"flow_m3s": 0.3, "total_pressure_pa": 250.0, "shaft_power_w": 110.0},
            {"flow_m3s": 0.5, "total_pressure_pa": 140.0, "shaft_power_w": 90.0},
        ],
        "extrapolation": "reject",
    }
    fan.update(fan_over)
    return {"duct": duct, "fan": fan}


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_operating_point_benchmark_end_to_end(bench_duct, bench_fan):
    r = client.post("/operating-point", json=bench_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    q_exact = bench_analytic_root(bench_duct)
    assert 0.3 < body["flow_m3s"] < 0.5
    assert body["flow_m3s"] == pytest.approx(q_exact, abs=1e-6)
    # 交点闭合：风机全压 ≈ 风管压升，残差接近 0
    assert body["residual_pa"] == pytest.approx(0.0, abs=1e-5)
    assert body["total_pressure_pa"] == pytest.approx(
        body["duct_required_pressure_pa"], abs=1e-5
    )
    assert body["fan_static_pressure_pa"] == pytest.approx(300.0)
    # 流速口径唯一
    assert body["velocity_ms"] == pytest.approx(body["flow_m3s"] / bench_duct.area_m2)


def test_speed_endpoint_recomputes_intersection():
    payload = bench_payload()
    payload["base_speed_rpm"] = 1000.0
    payload["new_speed_rpm"] = 1200.0
    r = client.post("/speed", json=payload)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["speed_ratio"] == pytest.approx(1.2)
    assert b["flow_ratio"] == pytest.approx(1.2, abs=1e-6)
    assert b["pressure_ratio"] == pytest.approx(1.44, abs=1e-5)
    assert b["power_ratio"] == pytest.approx(1.2**3, rel=1e-4)
    # 新点自身闭合
    assert b["new"]["residual_pa"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("diameter_m", 0.0),
        ("diameter_m", -0.2),
        ("length_m", -5.0),
        ("density", 0.0),
        ("density", -1.0),
        ("friction_factor", -0.01),
        ("local_loss_sum", -1.0),
    ],
)
def test_invalid_duct_inputs_are_422(field, value):
    payload = bench_payload()
    payload["duct"][field] = value
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json() or "issues" in r.json()


def test_empty_fan_curve_is_422():
    payload = bench_payload(samples=None)
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 422


def test_single_sample_is_422():
    payload = bench_payload(
        samples=[{"flow_m3s": 0.0, "total_pressure_pa": 300.0}]
    )
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 422


def test_both_samples_and_quadratic_is_422():
    payload = bench_payload()
    payload["fan"]["a2"] = -100.0
    payload["fan"]["a1"] = 0.0
    payload["fan"]["a0"] = 300.0
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 422


def test_quadratic_curve_endpoint():
    payload = bench_payload()
    payload["fan"] = {"a2": -100.0, "a1": 0.0, "a0": 400.0}
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["residual_pa"] == pytest.approx(0.0, abs=1e-5)


def test_out_of_range_samples_explicitly_rejected():
    # 管阻极弱：真正交点在 0.5 采样之外；reject 模式必须明确报错，不糊弄。
    payload = bench_payload()
    payload["duct"] = {
        "length_m": 1.0,
        "diameter_m": 1.0,
        "friction_factor": 0.01,
        "local_loss_sum": 0.0,
        "density": 1.204,
    }
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 422
    assert r.json()["error"] in {"NoIntersectionError", "ExtrapolationError"}


def test_declared_linear_extrapolation_accepted():
    payload = bench_payload(extrapolation="linear")
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 200


def test_invalid_speed_is_422():
    payload = bench_payload()
    payload["base_speed_rpm"] = 0.0
    payload["new_speed_rpm"] = 1000.0
    r = client.post("/speed", json=payload)
    assert r.status_code == 422


def test_density_change_direction_over_http():
    low = client.post("/operating-point", json=bench_payload()).json()
    high_payload = bench_payload()
    high_payload["duct"]["density"] = 3.0
    high = client.post("/operating-point", json=high_payload).json()
    assert high["flow_m3s"] < low["flow_m3s"]
