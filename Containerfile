# lightspeed-agentic-sandbox
#
# Multi-provider agent sandbox for OpenShift Lightspeed.
# Matches the production pod layout:
#   /app/skills  — skills mounted as OCI image volume
#   /tmp         — writable workspace for agent operations
#   /home/agent  — writable home directory

FROM registry.redhat.io/rhel9/python-312:latest

USER 0

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

# Enable EPEL for ripgrep
RUN dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

# Claude Code SDK requirements
# bash: Bash tool runs everything through it (included in base)
# git: Glob/Grep tools use git for repo detection and file listing
# ripgrep: Grep tool uses rg under the hood for fast search
# curl/wget: WebFetch and HTTP calls
# jq: JSON processing (essential for oc/kubectl output)
RUN dnf install -y --nodocs \
    bash \
    git \
    ripgrep \
    wget \
    jq \
    && dnf clean all

# SRE debugging toolkit — tools the agent uses when investigating cluster issues
# procps-ng: ps, top, free, uptime (process inspection)
# iproute: ip, ss (network inspection)
# bind-utils: dig, nslookup, host (DNS debugging)
# net-tools: netstat, ifconfig (legacy but widely used)
# openssl: certificate inspection, TLS debugging
# lsof: open file/socket inspection
# strace: syscall tracing
# tcpdump: network packet capture
# less/vim-minimal: file viewing/editing
# tree: directory visualization
# file: file type detection
# diffutils: diff (comparing configs)
# skopeo: copy/inspect container images between registries
# unzip/tar/gzip: archive handling
RUN dnf install -y --nodocs \
    procps-ng \
    iproute \
    bind-utils \
    net-tools \
    openssl \
    lsof \
    strace \
    tcpdump \
    less \
    vim-minimal \
    findutils \
    file \
    diffutils \
    skopeo \
    unzip \
    tar \
    gzip \
    && dnf clean all

# oc CLI for Kubernetes operations
ARG OC_VERSION=stable
ARG TARGETARCH=amd64
RUN curl -sL "https://mirror.openshift.com/pub/openshift-v4/${TARGETARCH}/clients/ocp/${OC_VERSION}/openshift-client-linux.tar.gz" | \
    tar -xz -C /usr/local/bin oc kubectl && \
    chmod +x /usr/local/bin/oc /usr/local/bin/kubectl

WORKDIR /app

# Claude Code CLI — required by claude-agent-sdk for tool execution
RUN dnf install -y --nodocs nodejs && dnf clean all && \
    npm install -g @anthropic-ai/claude-code && \
    claude --version

# Install Python package
COPY LICENSE /licenses/LICENSE
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev --extra all --extra eval

# dumb-init for proper signal propagation to child processes
RUN dnf install -y --nodocs dumb-init && dnf clean all

# RHEL Python base already has UID 1001 (default:root).
# Set up writable directories matching lightspeed-agent production layout.
RUN usermod -d /home/agent -l agent default && \
    mkdir -p /app/skills /tmp/agent-workspace /home/agent && \
    chown -R 1001:0 /app /home/agent /tmp/agent-workspace

ENV SHELL="/bin/bash"
ENV HOME="/home/agent"
ENV LIGHTSPEED_SKILLS_DIR="/app/skills"
ENV PATH="/app/.venv/bin:${PATH}"

USER 1001:1001

EXPOSE 8080

ENTRYPOINT ["dumb-init", "--"]
CMD ["python", "-m", "uvicorn", "lightspeed_agentic.app:app", "--host", "0.0.0.0", "--port", "8080"]

LABEL name="lightspeed-agentic-sandbox" \
      summary="Multi-provider agent sandbox for OpenShift Lightspeed" \
      description="Python agent with Claude, Gemini, OpenAI, and DeepAgents provider support"
