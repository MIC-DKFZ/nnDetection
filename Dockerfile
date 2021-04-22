#Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ARG env_det_num_threads=6
ARG env_det_verbose=1

# Setup environment variables
ENV TORCH_CUDA_ARCH_LIST=6.1+PTX;7.0+PTX;7.5+PTX FORCE_CUDA=1
ENV det_data=/opt/data det_models=/opt/models det_num_threads=$env_det_num_threads det_verbose=$env_det_verbose OMP_NUM_THREADS=1

# Install some tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y \
 git \
 cmake \
 make \
 wget \
 gnupg \
 build-essential \
 software-properties-common \
 gdb \
 ninja-build

# Setup miniconda and create a new python environment with python 3.7
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
  && chmod +x miniconda.sh \
  && ./miniconda.sh -b -p /opt/miniconda \
  && rm ./miniconda.sh \
  && ln -s /opt/miniconda/bin/activate /activate \
  && . /activate \
  && pip install numpy \
  && pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install own code
COPY ./requirements.txt .
RUN mkdir ${det_data} \
  && mkdir ${det_models} \
  && mkdir -p /opt/code/nndet \
  && . /activate \
  && pip install -r requirements.txt  \
  && pip install hydra-core --upgrade --pre \
  && pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git

WORKDIR /opt/code/nndet
COPY . .
RUN . /activate && pip install -v -e .
