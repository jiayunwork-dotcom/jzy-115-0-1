# 风机-风管工作点选型核算后端

单管 + 单台风机的**稳定运行工作点**求解服务。给定一段风管与一条风机特性曲线，
用括号法（bisection）解非线性方程

```
p_fan(Q) = p_duct(Q)
```

得到交点流量，回代得到工作点全压与管内流速；并按风机相似律在不同转速间
换算——转速变化时**整条风机曲线先变换、再与原风管重新求交**，绝不拿旧交点
流量直接乘转速比。

技术栈：Python 3.12 + FastAPI，单容器部署（`python:3.12-slim`）。
范围限定：不涉及多环管网平差，不含前端。

## 目录结构（按职责拆模块）

```
app/
  errors.py    # 领域异常（非法输入 / 越界 / 无交点），统一转 422
  duct.py      # 管阻：截面积、唯一流速口径 Q/A、动压、dP=K·p_dyn
  fan.py       # 风机曲线：采样点分段线性插值 + 外推策略，或二次曲线
  solver.py    # 括号法（倍步找括号 + 二分）求交点并回代工作点
  affinity.py  # 相似律：整条曲线变换 (rQ, r²p, r³P) 后重新求交
  schemas.py   # Pydantic 输入校验与响应模型（HTTP 层）
  routes.py    # HTTP 路由，只做序列化与调用领域模块
  main.py      # FastAPI 入口与统一错误处理
tests/         # pytest：逐条盯死物理关系与接口契约
examples/      # 可直接 curl 的请求样例
Dockerfile     # python:3.12-slim 单容器
```

## 物理模型

圆管、不可压缩、空气物性按 20 ℃ 固定（默认密度 ρ = 1.204 kg/m³）：

```
A       = π D² / 4
v       = Q / A                 # 全系统唯一的流速口径
p_dyn   = ρ v² / 2
K       = f L / D + Σζ
dP_duct = K · p_dyn             # ∝ Q²；Q=0 时为 0
```

风机曲线两种形式（二选一）：

- 采样点 `(Q, 全压[, 轴功率])`，点间线性插值；范围外默认 `extrapolation="reject"`
  明确拒绝；可选 `"linear"` 按端点斜率外推（事先声明，不默默给数）。
- 二次曲线 `p(Q) = a2 Q² + a1 Q + a0`，多项式全局成立，无采样域。

相似律（同一台风机，只变转速，r = n₂/n₁）：

```
Q₂/Q₁ = r，  p₂/p₁ = r²，  P₂/P₁ = r³
```

## 快速开始（Docker）

```bash
docker build -t fan-selection .
docker run --rm -p 8000:8000 fan-selection
# 自动文档： http://localhost:8000/docs
```

本地直接运行：

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口

单位（SI）：流量 m³/s，全压 Pa，流速 m/s，转速 rpm，轴功率 W。

### POST `/operating-point` —— 求当前工作点

```bash
curl -s -X POST localhost:8000/operating-point \
  -H 'Content-Type: application/json' \
  -d @examples/operating_point.json
```

响应关键字段：`flow_m3s`、`total_pressure_pa`、`velocity_ms`、
`duct_required_pressure_pa`、`residual_pa`（交点闭合误差，应≈0）、
`shaft_power_w`（提供了功率采样时）、`fan_static_pressure_pa`（Q=0 静压）。

### POST `/speed` —— 指定新转速，相似律换算后**重新求交**

```bash
curl -s -X POST localhost:8000/speed \
  -H 'Content-Type: application/json' \
  -d @examples/speed.json
```

返回 `base` 与 `new` 两个完整工作点，以及 `speed_ratio`、`flow_ratio`、
`pressure_ratio`、`power_ratio`。

### 错误行为

管径为零、管长为负、密度非正、摩阻/局部系数为负、风机曲线为空、
采样点重复、两种曲线形式混给、转速非正等，在**开始求解之前**就被挡下，
返回 HTTP 422：

```json
{"error": "RequestValidationError", "detail": "请求参数未通过校验", "issues": [ ... ]}
```

采样范围外且 `reject`、或范围内无交点：

```json
{"error": "ExtrapolationError", "detail": "流量 ... 超出风机采样范围 [...] ..."}
{"error": "NoIntersectionError", "detail": "...采样范围内不存在交点"}
```

## 预置手算算例（回归基准）

直管 + 内联风机（见 `examples/operating_point.json` 与 `tests/conftest.py`）：

- 风管：L=20 m，D=0.2 m，f=0.02，Σζ=1，ρ=1.204
  - A = π·0.2²/4 = 0.0314159 m²
  - K = 0.02·20/0.2 + 1 = **3**
  - dP = 0.5·ρ·K·(Q/A)² = c·Q²，c = 0.5·1.204·3/A² ≈ **1829.86**
- 风机采样：(0, 300)、(0.3, 250)、(0.5, 140) Pa。交点落在 0.3 与 0.5 之间，
  该段风机线：p = 250 + (140−250)/(0.5−0.3)·(Q−0.3) = **415 − 550Q**
- 求交：415 − 550Q = 1829.86·Q²

```
Q* = (-550 + √(550² + 4·1829.86·415)) / (2·1829.86)
   ≈ 0.34909 m³/s        （确认 0.3 < Q* < 0.5，与所用线段自洽）
p* = 415 − 550·0.34909 ≈ 223.00 Pa
v* = Q*/A ≈ 11.11 m/s
```

测试用独立解析公式（非求根器输出）比对该基准，防止求根器"自洽式错误"。

相似律手算（1000 → 1200 rpm，r=1.2）：

```
Q₂ ≈ 0.34909·1.2   ≈ 0.41891 m³/s
p₂ ≈ 223.00·1.44  ≈ 321.12 Pa
P₂/P₁ = 1.2³ = 1.728
```

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

58 个测试逐条覆盖：

- **Q² 律**：流量翻倍管阻≈四倍（`rel=1e-12`），三倍→九倍；Q=0 管阻归零；
- **流速唯一口径**：管阻中的 v 严格等于 Q/A；
- **加长/加粗糙/加局部阻力** → 交点流量下降，方向与两曲线形状一致；
- **密度调高** → 同流量管阻变大、交点流量减小；
- **Q=0** 风机给静压（关闭压），管阻为 0；
- **采样越界**：默认明确 422 拒绝；声明 `linear` 才外推；
- **相似律**：Q∝n、p∝n²、P∝n³，且新交点对**变换后的新曲线**和原风管
  双重闭合；旧交点搬到新转速下被验证为明显失衡（防止直接乘比例交差）；
- **非法输入**（管径 0、负管长、密度非正、空曲线、单点、重复流量、
  曲线形式混给、转速非正）在求解前全部 422；
- 基准算例端到端回归：交点必须在 0.3、0.5 两采样点之间并匹配解析根。
