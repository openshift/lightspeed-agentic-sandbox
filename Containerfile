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

# ---------------------------------------------------------------------------
# Builder stage: install Python deps from prefetched requirements
# ---------------------------------------------------------------------------
FROM registry.redhat.io/rhel9/python-312:latest AS builder

USER 0
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY requirements.*.txt ./

# Install Python packages from the platform-specific requirements file.
# In hermetic builds, Cachi2 sets PIP_* env vars pointing to prefetched deps.
# The unset avoids conflicts between Cachi2's --home and pip's --target.
RUN unset PIP_INSTALL_OPTIONS PIP_TARGET PIP_HOME PIP_PREFIX 2>/dev/null; \
    pip3.12 install --no-cache-dir --target /app/site-packages \
        -r requirements.$(uname -m).txt

# Install claude-code CLI.
# In hermetic builds (Konflux), cachi2 prefetches npm packages and sets the
# registry via cachi2.env — use npm ci against the lockfile.
# In non-hermetic builds (OpenShift BuildConfig), fall back to npm install -g.
COPY package.json package-lock.json ./
RUN dnf install -y --nodocs nodejs && dnf clean all
RUN if [ -f /cachi2/cachi2.env ]; then \
        . /cachi2/cachi2.env && \
        npm ci --ignore-scripts; \
    else \
        npm install -g @anthropic-ai/claude-code --ignore-scripts && \
        cp -a /usr/local/lib/node_modules /app/node_modules; \
    fi

# ---------------------------------------------------------------------------
# origincli stage: provides oc (kubectl is a symlink to oc in this image)
# ---------------------------------------------------------------------------
FROM registry.redhat.io/openshift4/ose-cli-rhel9:v4.21 AS origincli

# ---------------------------------------------------------------------------
# podman stage: provides catatonit init binary
# ---------------------------------------------------------------------------
FROM registry.redhat.io/ubi9/podman:9.8 AS podman

# ---------------------------------------------------------------------------
# Runtime stage: minimal image with only what the agent needs
# ---------------------------------------------------------------------------
FROM registry.redhat.io/rhel9/python-312:latest

USER 0
WORKDIR /app

# System packages (resolved from rpms.in.yaml via rpm prefetch).
# Split into functional groups for readability.

# Claude Code SDK requirements
RUN dnf install -y --nodocs \
    bash git wget jq \
    && dnf clean all

# SRE debugging toolkit
# tcpdump is installed separately with --setopt=strict=0 because it is not
# available in the UBI9 subset repos used by ci-operator builds (OpenShift CI).
# Konflux hermetic builds pre-fetch it via the RPM lockfile so the production
# image still includes tcpdump.
RUN dnf install -y --nodocs \
    procps-ng iproute bind-utils net-tools openssl \
    lsof strace \
    less vim-minimal findutils file diffutils \
    skopeo unzip tar gzip \
    && dnf install -y --nodocs --setopt=strict=0 tcpdump \
    && dnf clean all

# Node.js runtime (for claude-code CLI)
RUN dnf install -y --nodocs nodejs && dnf clean all

# Copy Python site-packages from builder
COPY --from=builder /app/site-packages /opt/app-root/lib64/python3.12/site-packages

# Copy claude-code npm installation from builder
COPY --from=builder /app/node_modules /app/node_modules
RUN ln -s /app/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude

# oc from the origincli stage; kubectl is a symlink to oc in that image
COPY --from=origincli /usr/bin/oc /usr/bin/oc
RUN ln -s /usr/bin/oc /usr/bin/kubectl

# catatonit init binary from the podman stage
COPY --from=podman /usr/libexec/podman/catatonit /usr/bin/catatonit

# Install generic-fetched binaries (ripgrep).
# In hermetic builds these are at /cachi2/output/deps/generic/.
# In non-hermetic builds fall back to fetching from the network.
COPY artifacts.lock.yaml ./
RUN ARCH=$(uname -m) && \
    GENERIC_DIR="/cachi2/output/deps/generic" && \
    if [ -d "$GENERIC_DIR" ]; then \
        tar -xzf "$GENERIC_DIR/ripgrep-${ARCH}.tar.gz" --strip-components=1 -C /usr/local/bin --wildcards '*/rg' && \
        chmod +x /usr/local/bin/rg; \
    else \
        echo "WARN: generic deps dir not found, fetching from network (non-hermetic)" && \
        RG_ARCH=$([ "$ARCH" = "aarch64" ] && echo "aarch64-unknown-linux-gnu" || echo "x86_64-unknown-linux-musl") && \
        curl -sL "https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-${RG_ARCH}.tar.gz" | \
            tar -xz --strip-components=1 -C /usr/local/bin --wildcards '*/rg' && \
        chmod +x /usr/local/bin/rg; \
    fi

# Copy application source and metadata
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml /app/README.md ./
COPY LICENSE /licenses/LICENSE

# RHEL Python base already has UID 1001 (default:root).
RUN usermod -d /home/agent -l agent default && \
    mkdir -p /app/skills /tmp/agent-workspace /home/agent && \
    chown -R 1001:0 /app /home/agent /tmp/agent-workspace

ENV SHELL="/bin/bash"
ENV HOME="/home/agent"
ENV LIGHTSPEED_SKILLS_DIR="/app/skills"
ENV PYTHONPATH="/app/src:/opt/app-root/lib64/python3.12/site-packages"
ENV PATH="/usr/local/bin:${PATH}"

USER 1001:1001

EXPOSE 8080

ENTRYPOINT ["/usr/bin/catatonit", "--"]
CMD ["python3.12", "-m", "uvicorn", "lightspeed_agentic.app:app", "--host", "0.0.0.0", "--port", "8080"]

LABEL name="lightspeed-agentic-sandbox" \
      summary="Multi-provider agent sandbox for OpenShift Lightspeed" \
      description="Python agent with Claude, Gemini, and OpenAI provider support"
