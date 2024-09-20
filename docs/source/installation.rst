Installation
============

The following sections provide all information needed to install nnDetection.

Prerequisites
-------------
nnDetection currently only supports execution on CUDA GPUs. Deployment to CPU devices is currently limited to a subset of the models and experimental (i.e. use at your own risk).
As a consequence, at least one NVIDIA GPU (compatible to with minimal PyTorch and CUDA versions is required).
While it is possible to run the planning stage of nnDetection for arbitrary GPUs via the `*EstV1` planners (e.g. `D3V002EstV1`) we recommend sticking with the new default `D3V002` which performs planning offline and requires a GPU with 16GB of VRAM. Scaling for larger GPUs needs to be performed manually for now (see some of our challenge participations for inspiration on scaling options).
A GPU with at least 11GB VRAM is highly recommended in all cases otherwise performance might suffer significantly.

Due to limited backwards compatibility of external dependencies, at least PyTorch 2.0 is required. Only CUDA versions >= 10.1 were tested during the development lifecycle of nnDetection.

Finally, at least Python 3.8 is needed to run nnDetection.

.. note::
  Requirements summary:
    - CUDA >= 10.1
    - Python >= 3.8
    - PyTorch >= 2.0

.. warning::
  nnDetection was developed on Linux => Windows is not supported. For usage on Windows systems, Docker is likely the best solution.

.. note::
  To get the best possible performance we recommend using CUDA 11.0+ with cuDNN 8.1.X+ and a (!)locally compiled version(!) of Pytorch 1.7.X-1.8.X
  Starting from PyTorch 1.9.X the pip installation gives mixed precision 3D conv speedup aswell.

Configuration
-------------

The configuration of the framework requires several environment variables to be set.

* `det_data`: Path to the source directory where all the data will be located
* `det_models`: Path to directory where all models will be saved
* `OMP_NUM_THREADS=1`: Needs to be set! Otherwise bad things will happen... Refer to batchgenerators documentation.
* `det_num_threads`: Number processes to use for augmentation (at least 6, default 12)

Optional flags for nnDetection:

* `nndet_eval_max_detections_image_based`: define number of predictions per image per class which is used for evaluation. Per default 400 is used.
* `det_verbose`: Can be used to deactivate progress bars (activated by default)
* `det_logging`: Specify the logging directory, by default logs will be written to the current training directory.
* `det_logger`: Define logger type. One of tensorboard | mlflow | none. nnDetection supports `MLFlow <https://www.mlflow.org/docs/latest/tracking.html>`_ or `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html?highlight=tensorboard>`_.

Only trensorboard is installed by default, other logger might require running additional isntallation instructions (e.g. via pip).

Source Install with Conda CUDA
------------------------------

This installation method provides a source install of nnDetection and uses CUDA installed via CONDA. It is recommended for most users to use this installation method due to its ease of use.

1. Install conda / miniconda etc. Please refer to the official repository for additional guidance.
2. Install setup dependencies
    - torch(https://pytorch.org/) (make sure to match the pytorch and CUDA versions!, Note minimal torch version from `Prerequisites`)
    - torchvision(https://github.com/pytorch/vision)(make sure to match the versions with pytorch!).
3. Install CUDA via conda, e.g. to install CUDA 11.7.1 `conda install cuda -c nvidia/label/cuda-11.7.1`
4. Set the CUDA_HOME path to your conda environment, e.g. `export CUDA_HOME=$CONDA_PREFIX`
5. [Optional] In some cases the local C/C++ compile version is not supported by the selected PyTorch version. In this cases it is possible to install them via conda as well, e.g. `conda install gxx_linux-64==9.3.0` and set the environment variables `export CXX=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++ && export CC=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-cc`.
6. Clone nnDetection, `cd [path_to_repo]` and `pip install -e .`
7. Set environment variables.
8. Test Installation
    - Test installation by running `python -c "import torch; import nndet._C; import nndet"` (not inside the pytorch folder)
    - To test the whole installation please run the Toy Data set example.


Summary of all steps (excluding env variables):

.. code-block:: bash

  # when writing this PyTorch CUDA 12.1 is the default version of PyTorch you might need to adapt!
  pip install torch torchvision
  conda install cuda -c nvidia/label/cuda-12.1.1 # adapt to the needed CUDA version from pytorch
  conda install gxx_linux-64 # specify version if needed
  export CUDA_HOME=$CONDA_PREFIX
  export CXX=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++
  export CC=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-cc
  git clone https://github.com/MIC-DKFZ/nnDetection
  cd nnDetection
  pip install -e . -v


Source Install with Local CUDA
------------------------------

This installation method provides the best performance but we  only recommend this for experiences users. Prefer Docker or Source with CONDA CUDA over this one.

1. Install CUDA (make sure to select compatible versions(https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
2. [Optional] Depending on your GPU you might need to set `TORCH_CUDA_ARCH_LIST`, check compute capabilities(https://developer.nvidia.com/cuda-gpus) here.
3. Install setup dependencies
    - torch(https://pytorch.org/) (make sure to match the pytorch and CUDA versions!, Note minimal torch version from `Prerequisites`)
    - torchvision(https://github.com/pytorch/vision)(make sure to match the versions with pytorch!).
4. Clone nnDetection, `cd [path_to_repo]` and `pip install -e .`
5. Set environment variables.
6. Test Installation
    - Test installation by running `python -c "import torch; import nndet._C; import nndet"` (not inside the pytorch folder)
    - To test the whole installation please run the Toy Data set example.


Docker
------

The easiest way to get started with nnDetection is the provided is to build a Docker Container with the provided Dockerfile.

Please install docker and nvidia-docker2(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) before continuing.

All projects which are based on nnDetection assume that the base image was built with the following tagging scheme `nnDetection:[version]`.
To build a container (nnDetection Version 0.1) run the following command from the base directory:

.. code-block:: bash

  docker build -t nndetection:0.1 --build-arg env_det_num_threads=6 --build-arg env_det_verbose=1 .

(`--build-arg env_det_num_threads=6` and `--build-arg env_det_verbose=1` are optional and are used to overwrite the provided default parameters)

The docker container expects data and models in its own `/opt/data` and `/opt/models` directories respectively.
The directories need to be mounted via docker `-v`. For simplicity and speed, the ENV variables `det_data` and `det_models` can be set in the host system to point to the desired directories.
Run to start the container:

.. code-block:: bash

  docker run --gpus all -v ${det_data}:/opt/data -v ${det_models}:/opt/models -it --shm-size=24gb nndetection:0.1 /bin/bash

.. warning::
  When running a training inside the container it is necessary to increase the shared memory(https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container) (via --shm-size).

Extended Dev Install
--------------------

For a full development installation of nnDetection, just run `pip install -e .\[dev\]`. This will install additional pckages which are required for full unittesting.
The remaining steps are the same as for any source intallation of nnDetection


Installation & Initial Setup Errors
-----------------------------------

Error: Undefined CUDA symbols when importing `nndet._C` or other import related Errors from `nndet._C` or CUDA related ARCH errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nnDetection includes additional CUDA code which needs to compiled upon installation and thus requires correct configuration of the CUDA dependencies.
Please double check CUDA version of your PC, pytorch, torchvision and nnDetection build.
This can be done by running `nndet_env` if the installation succeeded  or by running `python scripts/utils.py`.
An example output of the command is shown below:

.. note:: 

    ----- PyTorch Information -----
    PyTorch Version: 1.11.0+cu113
    PyTorch Debug: False
    PyTorch CUDA: 11.3
    PyTorch Backend cudnn: 8200
    PyTorch CUDA Arch List: ['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
    PyTorch Current Device Capability: (7, 5)
    PyTorch CUDA available: True

    ----- System Information -----
    System NVCC: nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2021 NVIDIA Corporation
    Built on Sun_Aug_15_21:14:11_PDT_2021
    Cuda compilation tools, release 11.4, V11.4.120
    Build cuda_11.4.r11.4/compiler.30300941_0

    System Arch List: None
    System OMP_NUM_THREADS: 1
    System CUDA_HOME is None: True
    System CPU Count: 8
    Python Version: 3.8.11 (default, Aug  3 2021, 15:09:35)
    [GCC 7.5.0]

    ----- nnDetection Information -----
    det_num_threads 6
    det_data is set True
    det_models is set True

Things to look out for:

Make sure that the versions of PyTorch CUDA and NVCC CUDA match (minor version mismatch as in this case, will work without error but could potentially introduce bugs.)
`OMP_NUM_THREADS` should always be set to 1 and `det_num_threads` should always be lower or equal `Systemm CPU Count`.
Make sure to delete the `build` folder before rerunning the installation since it won't recompile the code otherwise.

Error: No kernel image is available for execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You are probably executing the build on a machine with a GPU architecture which was not present/set during the build.

Please check [link](https://developer.nvidia.com/cuda-gpus) to find the correct SM architecture and set `TORCH_CUDA_ARCH_LIST`
approriately (e.g. check Dockefile for example).
As before make sure to delete the `build` folder when rerunning the installation process.

Error still persists
~~~~~~~~~~~~~~~~~~~~

Please open an Issue and provide your environment as obtained by `nndet_env`.

Training doesn't start or is stuck
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Please run `nndet_env` and make sure `OMP_NUM_THREADS` is set to 1. No other values are supported here. To increase the number of workers used for IO and augmentation adjust `nndet_num_threads`.
* Try running the training without multiprocessing as a sanity check: `nndet_train XXX -o augment_cfg.multiprocessing=False`. Don't use this for the full training, this is just one step of the debugging process.
* Please open an Issue and provide your environment as obtained by `nndet_env` and report if the training without multiprocessing started correctly.

GPU requirements
~~~~~~~~~~~~~~~~

nnDetection v0.1 was developed for GPUs with at least 11GB of VRAM (e.g. RTX2080TI, TITAN RTX).
All of our experiments were conducted with a RTX2080TI.
While the memory can be adjusted by manipulating the correct setting we recommend using the default values for now.
Future releases will refactor the planning stage to improve the VRAM estimation and add support for different memory budgets.
