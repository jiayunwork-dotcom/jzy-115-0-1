# 风机选型核算后端（单管 + 单台风机）

给定一段圆形风管与一台风机的特性曲线，求它们真正稳定运行的工作点（风机全压曲线
与风管阻力曲线的交点），并按风机相似律换算不同转速下的表现。

- Python 3.12 + FastAPI，纯 HTTP 接口，无前端
- 求根采用**括号法 + 二分法**，不依赖解析公式，采样曲线/拟合曲线通用
- 非法输入在进入求根前统一挡下（422 + 中文说明）
- 流量超出风机曲线声明范围时：**明确拒绝**或按**事先声明的**线性方式外推，二选一

## 物理模型

空气物性按 20 ℃固定（ρ = 1.2 kg/m³，可在请求中覆盖，但必须为正）。

圆管截面积

```
A = π D² / 4
```

流速全项目只有一个口径（`app/duct.py::velocity`），管阻与任何下游计算都从它取数：

```
v = Q / A
动压   pd = ρ v² / 2
管阻压升 Δp = (λ L/D + Σζ) · pd = K Q²
       K = (λ L/D + Σζ) · ρ / (2 A²)
```

- 管阻严格正比于 Q²：流量翻倍，压升约四倍
- Q = 0 时管阻压升为 0，风机在此点给出关断静压（动压为 0，全压 = 静压）

工作点方程（一般为非线性，二分求根）：

```
f(Q) = p_fan(Q) − Δp_duct(Q) = 0
```

相似律（同机、同系统、同密度；效率假设不变）：

```
Q₂/Q₁ = n₂/n₁     p₂/p₁ = (n₂/n₁)²     P₂/P₁ = (n₂/n₁)³
```

**转速一变，整条风机曲线整体变换后必须重新求交**（见 `app/affinity.py`），
新工作点绝不是旧交点乘个转速比；测试里有专门的 spy 盯住求根器被调用两次。

## 模块划分（按职责拆分）

| 模块 | 职责 |
|---|---|
| `app/duct.py` | 风管几何、流速（唯一口径）、动压、管阻压升 |
| `app/fan_curve.py` | 风机特性曲线：采样点插值 + 越界策略，或二次曲线 |
| `app/solver.py` | 括号法/二分法求工作点 |
| `app/affinity.py` | 曲线整体变换、相似律换算、重新求交 |
| `app/validation.py` | 领域对象的输入校验（在求解前拦截） |
| `app/schemas.py` | HTTP 请求/响应 Pydantic 模型 |
| `app/main.py` | FastAPI 路由、schema↔领域对象映射、错误响应 |

## 接口

### `POST /operating-point` — 求当前工作点

```json
{
  "duct": {
    "length_m": 10.0,
    "diameter_m": 0.2,
    "friction_factor": 0.02,
    "local_loss_coeff": 1.0,
    "density": 1.2
  },
  "fan": {
    "kind": "sampled",
    "flows_m3s": [0.0, 0.1, 0.2, 0.3],
    "pressures_pa": [80.0, 72.0, 64.0, 40.0],
    "extrapolation": "reject",
    "max_flow_m3s": 100.0
  }
}
```

风机也可以给二次曲线：`{"kind": "quadratic", "a2": -400, "a1": -20, "a0": 120}`。

返回：`flow_m3s`、`total_pressure_pa`（= `duct_pressure_pa`，二者闭合）、
`velocity_ms`、`iterations`、`fan_shutoff_pressure_pa`。

### `POST /affinity` — 指定新转速，曲线变换后重新求交

请求体在上面基础上增加：

```json
{ "base_speed_rpm": 1400.0, "new_speed_rpm": 2100.0, "fan_total_efficiency": 0.65 }
```

效率可选；给了才输出轴功率（否则只输出空气功率 Q·p）。
返回新旧两个工作点、转速比及流量/压力/功率比值、`re_solved: true`。

### 其他

- `GET /example`：预置的可手算算例请求体（回归基准）
- `GET /health`：存活探针
- `GET /docs`：Swagger UI（自动生成的 OpenAPI）

错误响应统一形如 `{"error": "...", "detail": "中文说明"}`：
非法输入 422（`invalid_input` / `invalid_request` / `fan_curve_error`），
物理上无工作点 409（`no_operating_point`）。

## 预置手算算例（回归基准）

风管 D = 0.2 m、L = 10 m、λ = 0.02、Σζ = 1.0、ρ = 1.2 kg/m³：

```
A = π·0.01 = 0.0314159 m²
Kζ = 0.02·10/0.2 + 1 = 2.0
K  = Kζ·ρ/(2A²) ≈ 1215.8542 Pa/(m³/s)²
```

风机采样点 (0, 80)、(0.1, 72)、(0.2, 64)、(0.3, 40)。交点位于
0.2~0.3 段，该段风机直线为 `p = 112 − 240 Q`。解

```
1215.8542 Q² + 240 Q − 112 = 0
Q* ≈ 0.22045 m³/s   （严格落在 0.2 与 0.3 两个采样点之间）
p* ≈ 59.091 Pa
v* = Q*/A ≈ 7.017 m/s
零流量静压 = 80 Pa
```

该基准用独立的求根公式（不经过二分器）写死在
`tests/conftest.py::benchmark_values` 与 `tests/test_solver.py` 中，
HTTP 层也有对应回归测试。

## 本地运行与测试

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest
```

## Docker（单容器）

```bash
docker build -t fan-selection-backend .
docker run --rm -p 8000:8000 fan-selection-backend
# 快速核对
curl -s localhost:8000/health
curl -s -X POST localhost:8000/operating-point \
  -H 'content-type: application/json' \
  -d "$(curl -s localhost:8000/example | python -c 'import sys,json;print(json.dumps({k:v for k,v in json.load(sys.stdin).items() if k in ("duct","fan")}))')"
```

## 范围边界

只解单管 + 单台风机的工作点；不做多环管网平差，不提供前端页面或下单流程。
风机曲线默认按同一进口密度给出，密度敏感性只体现在风管侧。
