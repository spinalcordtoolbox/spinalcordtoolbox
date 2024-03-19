FROM ubuntu:22.04

RUN apt-get update && apt-get install -y git curl bzip2 libglib2.0-0 libgl1-mesa-glx libxrender1 libxkbcommon-x11-0 libdbus-1-3 gcc
RUN git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git sct
RUN cd sct
RUN ./install_sct -y
RUN rource /root/.bashrc
