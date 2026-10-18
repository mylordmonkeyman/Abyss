FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG RUNNER_VERSION
ARG RUNNER_SHA256
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential cmake ninja-build pkg-config \
      python3 python3-venv python3-pip python3-numpy python3-yaml \
      numactl libnuma-dev libaio-dev liburing-dev unzip jq procps \
      cuda-command-line-tools-12-8 cuda-nvtx-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Copy the version and Linux-x64 SHA256 from GitHub's New self-hosted runner page.
# Registration credentials are supplied later, never as image-build arguments.
RUN test -n "$RUNNER_VERSION" && test -n "$RUNNER_SHA256" \
    && mkdir -p /opt/actions-runner-template \
    && curl --fail --location --retry 3 \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
      --output /tmp/runner.tar.gz \
    && echo "$RUNNER_SHA256  /tmp/runner.tar.gz" | sha256sum --check --strict \
    && tar -xzf /tmp/runner.tar.gz -C /opt/actions-runner-template \
    && rm /tmp/runner.tar.gz \
    && /opt/actions-runner-template/bin/installdependencies.sh \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1001 --shell /bin/bash runner \
    && mkdir -p /actions-runner \
    && chown runner:runner /actions-runner
COPY entrypoint.sh /usr/local/bin/abyss-runner-entrypoint
RUN chmod 755 /usr/local/bin/abyss-runner-entrypoint

ENV PATH="/usr/local/cuda-12.8/bin:${PATH}" \
    TMPDIR=/actions-runner/_tmp
USER runner
WORKDIR /actions-runner
ENTRYPOINT ["/usr/local/bin/abyss-runner-entrypoint"]
CMD ["./run.sh"]
