FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget unzip git bzip2
RUN wget https://github.com/neuropoly/spinalcordtoolbox/archive/master.zip
RUN unzip master.zip
WORKDIR spinalcordtoolbox-master
RUN ./install_sct -y
RUN echo "export PATH=/root/sct_dev/bin:$PATH" >>~/.bashrc
