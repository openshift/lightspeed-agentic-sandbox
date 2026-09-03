UV := uv

CONTAINER_RUNTIME := $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
IMAGE := lightspeed-agentic-sandbox:latest
SANDBOX_IMAGE ?= $(IMAGE)

ifneq ($(filter e2e,$(MAKECMDGOALS)),)
E2E_EXTRA_TARGETS := $(filter-out e2e,$(MAKECMDGOALS))
ifneq ($(E2E_EXTRA_TARGETS),)
.PHONY: $(E2E_EXTRA_TARGETS)
$(E2E_EXTRA_TARGETS):
	@:
endif
endif

.PHONY: install install-all lock test lint format mypy verify verify-hermetic-requirements e2e image clean help \
       requirements bump-deps rpm-lockfile konflux-requirements

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev dependencies via uv
	$(UV) sync --extra dev

install-all: ## Install all provider, dev, and e2e dependencies via uv
	$(UV) sync --all-extras

lock: ## Refresh uv.lock from pyproject.toml
	$(UV) lock

test: ## Run unit tests
	$(UV) run pytest tests/ -v --ignore=tests/e2e

lint: ## Run ruff linter
	$(UV) run ruff check .

format: ## Auto-format with ruff
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

mypy: ## Run mypy against application package
	$(UV) run mypy src/lightspeed_agentic

verify: verify-hermetic-requirements ## Run non-mutating formatting, lint, and type checks
	$(UV) run ruff format . --check
	$(UV) run ruff check .
	$(UV) run mypy src/lightspeed_agentic

verify-hermetic-requirements: ## Verify hermetic build hash files are in sync with uv.lock
	bash scripts/verify_hermetic_requirements.sh

image: ## Build container image for local development and e2e
	$(CONTAINER_RUNTIME) build -t $(IMAGE) .

e2e: image ## Batch cluster E2E BDD (make e2e openai-agents). Needs oc/KUBECONFIG; optional: E2E_ARGS, E2E_SKIP_FIXTURES=1.
	IMAGE="$(IMAGE)" SANDBOX_IMAGE="$(SANDBOX_IMAGE)" E2E_ARGS="$(E2E_ARGS)" bash scripts/e2e-containers.sh $(filter-out e2e,$(MAKECMDGOALS))

requirements: pyproject.toml ## Generate requirements.txt files for Konflux hermetic builds
	$(UV) pip compile pyproject.toml --extra all --extra e2e \
		-o requirements.x86_64.txt --generate-hashes \
		--python-platform x86_64-unknown-linux-gnu --upgrade
	$(UV) pip compile pyproject.toml --extra all --extra e2e \
		-o requirements.aarch64.txt --generate-hashes \
		--python-platform aarch64-unknown-linux-gnu --upgrade
	python3 scripts/gen-build-deps.py \
		requirements-build.txt \
		requirements.x86_64.txt requirements.aarch64.txt

konflux-requirements: ## Resolve RHOAI+PyPI deps for Konflux hermetic builds
	python3 scripts/konflux_resolve.py --profile cpu

bump-deps: ## Upgrade all dependencies and regenerate requirements
	$(UV) lock --upgrade
	$(MAKE) requirements

rpm-lockfile: .konflux/rpms.in.yaml .konflux/redhat.repo ## Regenerate rpms.lock.yaml (requires podman + RH subscription)
	./scripts/generate-rpm-lock.sh -a $${ACTIVATION_KEY} -g $${ORG_ID}

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info .venv .pytest_cache .mypy_cache .ruff_cache node_modules/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
