FROM python:3.10-slim-bullseye AS build

ARG SCT_TOOLBOX_VERSION

ENV SCT_TOOLBOX_VERSION=${SCT_TOOLBOX_VERSION:-"7.2"}
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# If build context is SCT repository, copy it all and overwrite the cloned version
ADD . /tmp/sct_build_context
WORKDIR /tmp/sct_build_context
RUN if [ -f setup.py ]; then \
        cp -r $PWD/* /tmp/spinalcordtoolbox/. ; \
    fi

# Install SCT and dependencies
WORKDIR /tmp/spinalcordtoolbox
RUN --mount=type=cache,target=/root/.conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    bash install_sct -y -c -g -p

# Package the conda environment for deployment
RUN export SCT_VERSION=$(cat /tmp/spinalcordtoolbox/spinalcordtoolbox/version.txt) && \
    export PATH="/root/sct_${SCT_VERSION}/python/bin:${PATH}" && \
    conda install -c conda-forge conda-pack && \
    cd /root/sct_${SCT_VERSION} && \
    conda-pack -p python/envs/venv_sct --compress-level 0 -j -1 --format tar && \
    mkdir /venv && tar -xf venv_sct.tar -C /venv

# Unpack the conda environment and set it up for use
WORKDIR /venv
RUN ./bin/conda-unpack 


FROM continuumio/miniconda3:24.11.1-0

# Copy SCT conda environment from build stage
COPY --from=build /venv /venv_sct
ENV PATH="/venv_sct/bin:${PATH}"


# Once build in validated, add non-root user and set locales

# Setup time and locales
# RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

# # Create non-root user for development
# RUN groupadd -r sctdev && useradd -r -g sctdev sctdev

# # Switch to non-root user
# USER sctdev
