FROM python:3.12-slim

# 不写 .pyc、日志不缓冲，容器内排障更直接
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先单独装依赖，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷应用代码（测试也一并打入，便于在容器里核对）
COPY app ./app
COPY pytest.ini ./pytest.ini
COPY tests ./tests

EXPOSE 8000

# 单容器直接对外服务；常驻运行，可挂到任意 ASGI 部署前面
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
