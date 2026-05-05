FROM python:3.10-slim-bullseye AS build

ARG SCT_TOOLBOX_VERSION

ENV SCT_TOOLBOX_VERSION=${SCT_TOOLBOX_VERSION:-"7.2"}
ENV CONDA_PKGS_DIRS=/root/.conda/pkgs \
    MINIFORGE_CACHE_DIR=/root/.cache/miniforge \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        git \
        bzip2 \
        ca-certificates \
        libdbus-1-3 \
        libglib2.0-0 \
        libxkbcommon-x11-0 \
        libxrender1

# Clone SCT from repository
ADD https://github.com/spinalcordtoolbox/spinalcordtoolbox.git#${SCT_TOOLBOX_VERSION} /tmp/spinalcordtoolbox

# Copy local build context and overwrite the cloned version
ADD setup.py \
    setup.cfg \
    README.rst \
    MANIFEST.in \
    requirements.txt \
    install_sct \
    LICENSE /tmp/sct_build_context/
ADD spinalcordtoolbox /tmp/sct_build_context/spinalcordtoolbox/
ADD data/ /tmp/sct_build_context/data/
ADD contrib/ /tmp/sct_build_context/contrib/

WORKDIR /tmp/sct_build_context
RUN if [ -f setup.py ]; then \
        cp -r $PWD/* /tmp/spinalcordtoolbox/. ; \
    fi

# Install SCT and dependencies
WORKDIR /tmp/spinalcordtoolbox
RUN --mount=type=cache,target=/root/.cache/miniforge,sharing=locked \
    --mount=type=cache,target=/root/.conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.mamba/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    bash install_sct -y -c -g -p

# Bind the installed SCT directory to a consistent location for later stages
RUN ln -s /root/sct_$(cat spinalcordtoolbox/version.txt) /root/sct
ENV SCT_DIR=/root/sct

FROM build AS package

ENV CONDA_PKGS_DIRS=/root/.conda/pkgs

# Package the conda environment for deployment
RUN --mount=type=cache,target=/root/.conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.mamba/pkgs,sharing=locked \
    export SCT_VERSION=$(cat /tmp/spinalcordtoolbox/spinalcordtoolbox/version.txt) && \
    export PATH="/root/sct_${SCT_VERSION}/python/bin:${PATH}" && \
    conda install -c conda-forge conda-pack && \
    cd /root/sct_${SCT_VERSION} && \
    conda-pack -p python/envs/venv_sct --compress-level 0 -j -1 --format tar && \
    mkdir /venv && tar -xf venv_sct.tar -C /venv

# Unpack the conda environment and set it up for use
WORKDIR /venv
RUN ./bin/conda-unpack

FROM continuumio/miniconda3:24.11.1-0

ARG SCT_DEEPSEG_MODELS
ENV SCT_DEEPSEG_MODELS=${SCT_DEEPSEG_MODELS:-""}

# Copy SCT conda environment from build stage
COPY --from=package /venv /venv_sct
ENV PATH="/venv_sct/bin:${PATH}"
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Install listed models if any (comma separated list, call sct_deepseg <model> -install for each)
RUN if [ -n "$SCT_DEEPSEG_MODELS" ]; then \
        echo "Installing deepseg models: $SCT_DEEPSEG_MODELS"; \
        printf '%s\n' "$SCT_DEEPSEG_MODELS" | tr ',' '\n' | while read -r MODEL; do \
            echo "Installing model: $MODEL"; \
            sct_deepseg "$MODEL" -install; \
            # Sleep for 5 seconds to avoid potential rate limits
            sleep 5; \
        done; \
    fi

# Once build is validated, add non-root user and set locales

# Setup time and locales
# RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

# # Create non-root user for development
# RUN groupadd -r sctdev && useradd -r -g sctdev sctdev

# # Switch to non-root user
# USER sctdev
