FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
ARG SOURCE_REVISION=working-tree
LABEL org.opencontainers.image.title="GeoDistill training" \
      org.opencontainers.image.revision=$SOURCE_REVISION
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch \
    RCLONE_CONFIG=/secrets/rclone.conf MPLCONFIGDIR=/cache/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates rclone tini openssh-server curl wget python3-venv && \
    rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* && mkdir -p /run/sshd /data /runs /cache /config /secrets
RUN python -m venv --system-site-packages /opt/train-venv
ENV PATH="/opt/train-venv/bin:$PATH"
WORKDIR /app
COPY docker/requirements-train.txt /tmp/requirements-train.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-train.txt && \
    python -m pip check && python -m pip freeze > /opt/python-packages.txt
COPY README.md /app/README.md
COPY src /app/src
COPY configs /app/configs
COPY tests /app/tests
COPY scripts /app/scripts
COPY docker /app/docker
RUN python -m compileall -q /app/src && \
    chmod +x /app/docker/*.sh && \
    python -c "from src.runtime.__main__ import source_hash; print(source_hash())" > /opt/source.sha256
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "python", "-m", "src.runtime"]
CMD ["--help"]
