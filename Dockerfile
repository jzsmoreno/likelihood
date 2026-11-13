# syntax=docker/dockerfile:1
#
# Likelihood model catalog.
#
# The server introspects the package sources with `model_catalog.py` at runtime,
# so the repository is copied into the image (there is no build step) and the
# catalog is re-introspected whenever a model file changes (`LM_CATALOG_TTL`).
#
# Build:
#   docker build -t likelihood-catalog .
#   docker run --rm -p 8000:8000 likelihood-catalog
#
# Models that need TensorFlow, PyTorch, Keras Tuner or scikit-learn are optional:
#   docker build --build-arg INSTALL_FULL_EXTRAS=true -t likelihood-catalog:full .

FROM python:3.12-slim AS runtime

# Install the optional `[full]` extras (tensorflow, torch, keras-tuner, ...).
ARG INSTALL_FULL_EXTRAS=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/likelihood \
    LM_HOST=0.0.0.0 \
    LM_PORT=8000

# graphviz: renders the pydot/graphviz diagrams some models produce.
# libgomp1: OpenMP runtime used by the numpy/scipy/torch wheels.
RUN apt-get update \
    && apt-get install --no-install-recommends -y graphviz libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is reused while only the sources change.
# `requirements.txt` is the project's single dependency list, so installing it
# verbatim keeps the image in sync with `setup.py` exactly as
# `pip install "likelihood[server]"` would (it also carries the lint tooling).
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" \
    && if [ "${INSTALL_FULL_EXTRAS}" = "true" ]; then \
        python -m pip install \
            "networkx" \
            "pyvis" \
            "tensorflow>=2.15.0" \
            "keras-tuner" \
            "scikit-learn" \
            "torch"; \
    fi

# Unprivileged account: the catalog only reads the sources and the playground
# runs model code in child processes owned by the same user.
RUN groupadd --system --gid 1001 likelihood \
    && useradd --system --uid 1001 --gid likelihood --create-home likelihood

COPY --chown=likelihood:likelihood . .

# Fail the build early when the server or the introspection layer is broken.
RUN python -c "import server_core, server_fastapi; print('catalog models:', len(server_core.CATALOG))"

USER likelihood

EXPOSE 8000

# Anonymous probes only need the liveness fields, so this also works with LM_API_TOKEN.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import json,os,urllib.request; port=os.environ.get('LM_PORT','8000'); r=urllib.request.urlopen('http://127.0.0.1:'+port+'/api/health', timeout=4); raise SystemExit(0 if json.load(r).get('status')=='ok' else 1)"

# LM_HOST / LM_PORT are the defaults of the CLI, so `-e LM_PORT=9000` works.
CMD ["python", "server_fastapi.py"]
