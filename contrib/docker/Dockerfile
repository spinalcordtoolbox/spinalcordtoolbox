# Use a the default Ubuntu 22.04 image
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update \
    && apt-get install -y \
        bzip2 \
        curl \
        gcc \
        git \
        libdbus-1-3 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libxkbcommon-x11-0 \
        libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install version 7.0 (for example) of SCT
RUN git clone \
      --branch 7.0 \
      https://github.com/spinalcordtoolbox/spinalcordtoolbox.git \
      /root/sct \
    && /root/sct/install_sct -y

# Add SCT to PATH
ENV PATH="/root/sct/bin:$PATH"

# Automatically activate SCT env in bash
RUN echo "source /root/sct/python/etc/profile.d/conda.sh && conda activate venv_sct" >> /root/.bashrc
