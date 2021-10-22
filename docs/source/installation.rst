Installation
============

TODOS
-----
- Python req.
- FOCE_CUDA
- FAQ
- Common Problems

  - clear cache (delete build)
  - nndet_env / collect env to check cuda versions)


Configuration
-------------
* `det_data`
* `det_models`
* `OMP_NUM_THREADS=1`
* `det_num_threads`
* `det_verbose`
* `det_logging`

Pypi
----

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


Source
------


User Install
~~~~~~~~~~~~
.. note::
  Requirements:
    - CUDA >= 10.1
    - Python >= 3.8
    - PyTorch >= 1.7

.. note::
  To get the best possible performance we recommend using CUDA 11.0+ with cuDNN 8.1.X+ and a (!)locally compiled version(!) of Pytorch 1.7.X-1.8.X
  Starting from PyTorch 1.9.X the pip installation gives mixed precision 3D conv speedup aswell.


1. Install CUDA (make sure to select compatible versions(https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
2. [Optional] Depending on your GPU you might need to set `TORCH_CUDA_ARCH_LIST`, check compute capabilities(https://developer.nvidia.com/cuda-gpus) here.
3. Install setup dependencies
    - torch(https://pytorch.org/) (make sure to match the pytorch and CUDA versions!)
    - torchvision(https://github.com/pytorch/vision)(make sure to match the versions with pytorch!).
4. Clone nnDetection, `cd [path_to_repo]` and `pip install -e .`
5. Set environment variables:
    - `det_data`: [required] Path to the source directory where all the data will be located
    - `det_models`: [required] Path to directory where all models will be saved
    - `OMP_NUM_THREADS=1` : [required] Needs to be set! Otherwise bad things will happen... Refer to batchgenerators documentation.
    - other configuration variables can be found above.
6. Test Instalation
    - Test installation by running `python -c "import torch; import nndet._C; import nndet"` (not inside the pytorch folder)
    - To test the whole installation please run the Toy Data set example.

.. warning::
  nnDetection was developed on Linux => Windows is not supported.


Developer Install
~~~~~~~~~~~~~~~~~
