# Linux x86_64; the host NVIDIA driver is injected by NVIDIA Container Toolkit.
FROM ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986 AS tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential pkg-config python3 uuid-dev libnl-3-dev libbsd0 libmd0 \
    && rm -rf /var/lib/apt/lists/*
COPY experiments/criu/build.sh /tmp/build-tools.sh
# build.sh fixes the CRIU and NVIDIA helper commits and records package hashes.
RUN apt-get update && (bash /tmp/build-tools.sh /opt/tools \
    || { tail -80 /opt/tools/build.log; exit 1; })

FROM ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv sudo ca-certificates libbsd0 libmd0 libgomp1 libuuid1 libnl-3-200 \
    procps iproute2 util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY experiments/finetuning/requirements.lock /tmp/requirements.lock
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.lock
COPY --from=tools /opt/tools/criu-4.2.1/criu/criu /opt/tools/criu-4.2.1/criu/criu
COPY --from=tools /opt/tools/criu-4.2.1/plugins/cuda/ /opt/tools/criu-4.2.1/plugins/cuda/
COPY --from=tools /opt/tools/criu-deps/usr/lib/x86_64-linux-gnu/ /opt/tools/criu-deps/usr/lib/x86_64-linux-gnu/
COPY --from=tools /opt/tools/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /opt/tools/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint
COPY --from=tools /opt/tools/package-hashes.txt /opt/tools/package-hashes.txt
RUN useradd --uid 1000 --create-home --shell /bin/bash learner \
    && printf 'learner ALL=(root) NOPASSWD: ALL\n' > /etc/sudoers.d/inference \
    && chmod 0440 /etc/sudoers.d/inference && chown -R learner:learner /opt/tools
WORKDIR /opt/experiment
COPY experiments/ experiments/
COPY tests/ tests/
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN PYTHONPATH=experiments/inference:experiments/finetuning:experiments/criu:experiments/cpu \
    python -m unittest discover -s tests -v
USER learner
CMD ["bash"]
