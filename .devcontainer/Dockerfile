# ===== Base image =====
FROM python:3.10-slim

# ===== Set working directory =====
WORKDIR /app

# ===== Install system dependencies (needed for numpy, xgboost, scikit-learn, matplotlib) =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ===== Copy project files =====
COPY code.py test_code.py Makefile requirements.txt ./

# ===== Install Python dependencies =====
# requirements.txt should have your core libs: polars, numpy, scikit-learn, xgboost, matplotlib
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir black flake8 pytest

# ===== Default command =====
# By default, run tests (good for CI)
CMD ["make", "check"]
