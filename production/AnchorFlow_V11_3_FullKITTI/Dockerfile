# CUDA *devel* is required because the metric teacher has a real CUDA extension.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel@sha256:b574d4ccf6d8856a5d87dcadc667aa4f95dc18d337ef3a28d02b7b01897d7081
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates tini && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip check
COPY . /app
# Explicit architectures support Ampere/Ada/Hopper; set 12.0 for a Blackwell GPU.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=2
RUN python bootstrap_teachers.py --config config.json && python -m pip freeze > /app/environment-freeze.txt
# BpOps --inplace stays in an isolated compatibility build directory.
ENV PYTHONPATH=/opt/teacher_sources/BpOps_build_torch210
ENTRYPOINT ["/usr/bin/tini","-g","--","python","-u","/app/pipeline.py"]
CMD ["all","--config","/config/config.json"]
