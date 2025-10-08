FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    METAMON_APP_DIR=/app \
    METAMON_CACHE_DIR=/cache \
    SHOWDOWN_HOST=127.0.0.1 \
    HF_HOME=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NODE_ENV=production

ARG TORCH_VERSION=2.5.1
ARG AMAGO_REPO_URL=https://github.com/UT-Austin-RPL/amago
ARG AMAGO_REF=main
ARG METAMON_REPO_URL=https://github.com/nivaldoh/metamon.git
ARG METAMON_REF=main

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      curl \
      git \
      gnupg \
      tini; \
    rm -rf /var/lib/apt/lists/*

# Install Node.js 22.x (Active LTS) from NodeSource
RUN set -eux; \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -; \
    apt-get update; \
    apt-get install -y --no-install-recommends nodejs; \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${METAMON_APP_DIR}

# Optionally clone the Metamon repo (with submodules) if build context
# did not provide the source tree. Local source copied later will overlay.
RUN set -eux; \
    if [ ! -f "${METAMON_APP_DIR}/pyproject.toml" ]; then \
      git config --global url."https://github.com/".insteadOf "git@github.com:"; \
      git config --global url."https://".insteadOf "git://"; \
      git clone --depth 1 --recurse-submodules --shallow-submodules --branch "${METAMON_REF}" "${METAMON_REPO_URL}" "${METAMON_APP_DIR}"; \
    fi

COPY . ${METAMON_APP_DIR}

RUN set -eux; \
    install -m 755 docker/with_showdown.sh /usr/local/bin/with-showdown; \
    install -m 755 docker/run_self_play.sh /usr/local/bin/run-self-play

# Clone amago if the source tree was provided without it
RUN set -eux; \
    if [ ! -f amago/setup.py ] && [ ! -f amago/pyproject.toml ]; then \
      git clone --depth 1 --branch "${AMAGO_REF}" "${AMAGO_REPO_URL}" amago; \
    fi

RUN set -eux; \
    python -m pip install --upgrade pip; \
    python -m pip install "torch==${TORCH_VERSION}"; \
    python -m pip install --no-build-isolation -e .; \
    python -m pip install --no-build-isolation -e ./amago

RUN set -eux; \
    cd server/pokemon-showdown; \
    NODE_ENV=development NPM_CONFIG_PRODUCTION=false npm ci; \
    npm run build --if-present || node build; \
    test -f dist/lib/static-server.js; \
    (npm prune --omit=dev || true); \
    npm cache clean --force

RUN set -eux; \
    useradd -m -u 10001 app; \
    mkdir -p "${METAMON_CACHE_DIR}"; \
    chown -R app:app "${METAMON_APP_DIR}" "${METAMON_CACHE_DIR}"

USER app

ENTRYPOINT ["/usr/bin/tini", "--", "with-showdown"]
CMD ["run-self-play"]
