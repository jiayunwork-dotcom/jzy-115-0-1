"""HTTP 层端到端测试（FastAPI TestClient，不依赖外部网络）。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app, EXAMPLE_REQUEST

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_benchmark_example_endpoint_and_regression(
    example_payload: dict, benchmark_values: dict
) -> None:
    # /example 返回的预置算例可以直接打 /operating-point。
    ex = client.get("/example").json()
    assert ex["duct"] == example_payload["duct"]
    assert ex["fan"] == example_payload["fan"]

    r = client.post("/operating-point", json=example_payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.2 < body["flow_m3s"] < 0.3
    assert body["flow_m3s"] == pytest.approx(benchmark_values["flow_m3s"], rel=1e-6)
    assert body["total_pressure_pa"] == pytest.approx(
        benchmark_values["pressure_pa"], rel=1e-6
    )
    assert body["velocity_ms"] == pytest.approx(benchmark_values["velocity_ms"], rel=1e-6)
    assert body["fan_shutoff_pressure_pa"] == 80.0
    assert body["total_pressure_pa"] == pytest.approx(
        body["duct_pressure_pa"], abs=1e-6
    )


def test_http_invalid_inputs_return_explained_422() -> None:
    cases = [
        # 管径为零
        {"duct": {"length_m": 10.0, "diameter_m": 0.0},
         "fan": {"kind": "quadratic", "a2": -100, "a1": 0, "a0": 80}},
        # 管长为负
        {"duct": {"length_m": -5.0, "diameter_m": 0.2},
         "fan": {"kind": "quadratic", "a2": -100, "a1": 0, "a0": 80}},
        # 密度非正
        {"duct": {"length_m": 10.0, "diameter_m": 0.2, "density": 0.0},
         "fan": {"kind": "quadratic", "a2": -100, "a1": 0, "a0": 80}},
    ]
    for payload in cases:
        r = client.post("/operating-point", json=payload)
        assert r.status_code == 422, payload
        body = r.json()
        assert "detail" in body and body["detail"]


def test_http_empty_fan_samples_rejected() -> None:
    payload = {
        "duct": {"length_m": 10.0, "diameter_m": 0.2},
        "fan": {
            "kind": "sampled",
            "flows_m3s": [],
            "pressures_pa": [],
            "extrapolation": "reject",
        },
    }
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 422


def test_http_malformed_body_422() -> None:
    r = client.post("/operating-point", json={"duct": {"length_m": 10.0}})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_http_out_of_range_rejection_422() -> None:
    payload = {
        "duct": {"length_m": 10.0, "diameter_m": 0.2,
                 "friction_factor": 0.02, "local_loss_coeff": 1.0},
        "fan": {
            "kind": "sampled",
            "flows_m3s": [0.0, 0.1, 0.2, 0.3],
            "pressures_pa": [1000.0, 950.0, 900.0, 850.0],
            "extrapolation": "reject",
        },
    }
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 422
    assert r.json()["error"] == "fan_curve_error"


def test_http_declared_linear_extrapolation_succeeds() -> None:
    payload = {
        "duct": {"length_m": 10.0, "diameter_m": 0.2,
                 "friction_factor": 0.02, "local_loss_coeff": 1.0},
        "fan": {
            "kind": "sampled",
            "flows_m3s": [0.0, 0.1],
            "pressures_pa": [300.0, 290.0],
            "extrapolation": "linear",
        },
    }
    r = client.post("/operating-point", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["flow_m3s"] > 0.1


def test_http_affinity_endpoint(example_payload: dict) -> None:
    payload = {**example_payload, "base_speed_rpm": 1400.0, "new_speed_rpm": 2100.0,
              "fan_total_efficiency": 0.65}
    # 基准采样域在升速后不够宽，给它一个更宽的曲线。
    payload["fan"] = {
        "kind": "sampled",
        "flows_m3s": [0.0, 0.1, 0.2, 0.3, 0.6, 0.7],
        "pressures_pa": [80.0, 72.0, 64.0, 40.0, -200.0, -300.0],
        "extrapolation": "reject",
        "max_flow_m3s": 100.0,
    }
    r = client.post("/affinity", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["re_solved"] is True
    r_ratio = 1.5
    assert body["flow_ratio"] == pytest.approx(r_ratio, abs=1e-6)
    assert body["pressure_ratio"] == pytest.approx(r_ratio**2, abs=1e-5)
    assert body["air_power"]["ratio"] == pytest.approx(r_ratio**3, abs=1e-5)
    assert body["shaft_power"]["ratio"] == pytest.approx(r_ratio**3, abs=1e-5)


def test_http_affinity_bad_speed_422(example_payload: dict) -> None:
    payload = {**example_payload, "base_speed_rpm": 0.0, "new_speed_rpm": 1500.0}
    r = client.post("/affinity", json=payload)
    assert r.status_code == 422


def test_example_payload_is_kept_in_sync() -> None:
    """回归基准描述与实际请求体保持一致。"""
    assert "直管" in EXAMPLE_REQUEST["description"]
