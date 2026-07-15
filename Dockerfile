# --- Stage 1: train the model inside the build (simple for project 1;
# in later projects the model artifact will come from a registry instead)
FROM python:3.12-slim AS trainer

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY app/ app/
RUN python scripts/train.py

# --- Stage 2: runtime image (no training deps beyond what inference needs)
FROM python:3.12-slim

# Run as non-root (k8s best practice). Numeric UID pinned explicitly —
# some kubelet/containerd combos can't reliably resolve a username to a
# UID during the runAsNonRoot check and fail closed (CreateContainerConfigError).
RUN useradd --uid 1000 --create-home appuser
WORKDIR /home/appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=trainer /build/app/ app/
USER 1000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
