# lightspeed-agentic-sandbox
#
# Multi-provider agent sandbox for OpenShift Lightspeed.
# Matches the production pod layout:
#   /app/skills  — skills mounted as OCI image volume
#   /tmp         — writable workspace for agent operations
#   /home/agent  — writable home directory
#
# Hermetic build: all dependencies are prefetched by Konflux/Hermeto.
# Network access is disabled during build.
#
# Base images are parameterized via build args:
#   Konflux (hermetic): overrides via .konflux/build.args → RHOAI 3.5 CPU images
#   OpenShift CI / development (non-hermetic): defaults below → UBI Python images
ARG BUILDER_BASE_IMAGE=registry.redhat.io/ubi9/python-312:latest
ARG RUNTIME_BASE_IMAGE=registry.redhat.io/ubi9/python-312-minimal:latest
ARG RUNTIME_DNF_COMMAND=microdnf

# ---------------------------------------------------------------------------
# Builder stage: install Python deps from prefetched requirements
# ---------------------------------------------------------------------------
FROM ${BUILDER_BASE_IMAGE} AS builder

USER 0
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY .konflux/requirements.hashes.*.txt .konflux/requirements.hermetic.txt ./

# Install Python packages from RHOAI wheels + PyPI sdist.
# In hermetic builds, Konflux injects PIP_* env vars via buildah secret mounts.
# The sed strips --index-url lines because uv rejects multiple index URLs
# when using --no-index --find-links.
ARG HERMETIC_BUILD=false
RUN if [ "${HERMETIC_BUILD}" = "true" ]; then \
        pip3.12 install --no-cache-dir uv && \
        uv venv && \
        for f in requirements.hashes.wheel.txt requirements.hashes.source.txt requirements.hashes.wheel.pypi.txt; do \
            sed -i '/^--index-url /d' "$f"; \
        done && \
        uv pip install --python .venv/bin/python --no-cache --no-index \
            --find-links ${PIP_FIND_LINKS} --no-deps \
            -r requirements.hashes.wheel.txt \
            -r requirements.hashes.source.txt \
            -r requirements.hashes.wheel.pypi.txt; \
    else \
        pip3.12 install --no-cache-dir uv && \
        uv sync --extra all --no-dev --locked; \
    fi

# ---------------------------------------------------------------------------
# origincli stage: provides oc (kubectl is a symlink to oc in this image)
# ---------------------------------------------------------------------------
# FROM registry.redhat.io/openshift4/ose-cli-rhel9:latest AS origincli

# ---------------------------------------------------------------------------
# podman stage: provides catatonit init binary
# ---------------------------------------------------------------------------
FROM registry.redhat.io/ubi9/podman:9.8 AS podman

# ---------------------------------------------------------------------------
# Runtime stage: minimal image with only what the agent needs
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE_IMAGE}

ARG BUILD_VERSION=unknown
ARG RUNTIME_DNF_COMMAND=microdnf

ENV LIGHTSPEED_BUILD_VERSION=${BUILD_VERSION}
LABEL org.opencontainers.image.revision=${BUILD_VERSION}

USER 0
WORKDIR /app

# System packages (resolved from .konflux/rpms.in.yaml via rpm prefetch).
# Split into functional groups for readability.

# Agent runtime requirements
RUN ${RUNTIME_DNF_COMMAND} install -y --nodocs \
    bash git wget jq \
    && ${RUNTIME_DNF_COMMAND} update -y --nodocs \
    && ${RUNTIME_DNF_COMMAND} clean all

# SRE debugging toolkit
# tcpdump is installed separately with || true because it is not
# available in the UBI9 subset repos used by ci-operator builds (OpenShift CI).
# Konflux hermetic builds pre-fetch it via the RPM lockfile so the production
# image still includes tcpdump.
RUN ${RUNTIME_DNF_COMMAND} install -y --nodocs \
    procps-ng iproute bind-utils net-tools openssl \
    lsof strace \
    less vim-minimal findutils file diffutils \
    skopeo unzip tar gzip \
    && (${RUNTIME_DNF_COMMAND} install -y --nodocs tcpdump || true) \
    && ${RUNTIME_DNF_COMMAND} clean all

# Copy Python site-packages from builder
COPY --from=builder /app/.venv/lib/python3.12/site-packages /opt/app-root/lib64/python3.12/site-packages

# oc from the origincli stage; kubectl is a symlink to oc in that image
# COPY --from=origincli /usr/bin/oc /usr/bin/oc
# RUN ln -s /usr/bin/oc /usr/bin/kubectl

# catatonit init binary from the podman stage
COPY --from=podman /usr/libexec/podman/catatonit /usr/bin/catatonit


# Copy application source outside /app so the agent workspace stays clean.
# The agent's Filesystem/Shell capabilities can see /app/; keeping source
# elsewhere prevents the LLM from reading sandbox internals during analysis.
# Intentionally root-owned and world-readable: the agent user should not modify app code.
COPY --from=builder /app/src /opt/lightspeed/src
COPY --from=builder /app/pyproject.toml /app/README.md /opt/lightspeed/
COPY LICENSE /licenses/LICENSE

RUN mkdir -p /app/skills /tmp/agent-workspace /home/agent && \
    chown -R 1001:0 /app /home/agent /tmp/agent-workspace

ENV SHELL="/bin/bash"
ENV HOME="/home/agent"
ENV LIGHTSPEED_SKILLS_DIR="/app/skills"
ENV PYTHONPATH="/opt/lightspeed/src:/opt/app-root/lib64/python3.12/site-packages"
ENV PATH="/usr/local/bin:${PATH}"

USER 1001:1001

# One-shot batch: no HTTP server and no long-lived process to probe. Readiness runs
# in-process at startup (readiness.py). Kubernetes probes are defined on the Pod
# by the operator, not via image HEALTHCHECK.
HEALTHCHECK NONE

ENTRYPOINT ["/usr/bin/catatonit", "--"]
CMD ["python3.12", "-m", "lightspeed_agentic.batch"]

LABEL name="openshift-lightspeed/lightspeed-agentic-sandbox-rhel9" \
      summary="Multi-provider agent sandbox for OpenShift Lightspeed" \
      description="Python agent with DeepAgents, Gemini, and OpenAI provider support" \
      cpe="cpe:/a:redhat:openshift_lightspeed:2::el9" \
      com.redhat.component="openshift-lightspeed" \
      io.k8s.display-name="OpenShift Lightspeed Agentic Sandbox" \
      io.k8s.description="Python agent with DeepAgents, Gemini, and OpenAI provider support" \
      io.openshift.tags="openshift-lightspeed,ols" \
      konflux.additional-tags="latest"
