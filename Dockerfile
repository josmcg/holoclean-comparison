FROM ubuntu:bionic
RUN apt-get update
RUN  apt-get install -y  wget gcc
RUN apt-get install -y software-properties-common 
RUN mkdir /holoclean
RUN wget -q  https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -O anaconda.sh
RUN bash anaconda.sh -b -p ~/.conda && rm anaconda.sh
ENV PATH /root/.conda/bin:$PATH
RUN conda create -n holo_env python=2.7
COPY . /holoclean
WORKDIR /holoclean
RUN /bin/bash -c "source activate holo_env;\
	pip install -r requirements.txt"

WORKDIR /vol

