import os
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1])
                )
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


def clean():
    """Custom clean command to tidy up the project root."""
    os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz")


def get_extensions():
    """
    Adapted from https://github.com/pytorch/vision/blob/master/setup.py
    and https://github.com/facebookresearch/detectron2/blob/master/setup.py
    """
    print("Build csrc")
    print("Building with {}".format(sys.version_info))

    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    extensions_dir = this_dir / "nndet" / "csrc"

    main_file = list(extensions_dir.glob("*.cpp"))
    source_cpu = (
        []
    )  # list((extensions_dir/'cpu').glob('*.cpp')) temporary until I added header files ...
    source_cuda = list((extensions_dir / "cuda").glob("*.cu"))
    print("main_file {}".format(main_file))
    print("source_cpu {}".format(source_cpu))
    print("source_cuda {}".format(source_cuda))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []
    extra_compile_args = {"cxx": []}

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        print("Adding CUDA csrc to build")
        print("CUDA ARCH {}".format(os.getenv("TORCH_CUDA_ARCH_LIST")))
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [str(extensions_dir)]

    ext_modules = [
        extension(
            "nndet._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


_SRC = os.path.dirname(__file__)
_REQ = os.path.join(_SRC, "requirements")

base_req = resolve_requirements(os.path.join(_REQ, "base.txt"))
extras = {
    "dev": resolve_requirements(os.path.join(_REQ, "dev.txt")),
}
readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))

setup(
    name="nndet",
    version="0.1",
    packages=find_packages(),
    # include_package_data=True,
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=base_req,
    python_requires=">=3.8",
    author="Division of Medical Image Computing, German Cancer Research Center",
    maintainer_email="m.baumgartner@dkfz-heidelberg.de",
    ext_modules=get_extensions(),
    extras_require=extras,
    cmdclass={
        "build_ext": BuildExtension,
        "clean": clean,
    },
    entry_points={
        "console_scripts": [
            "nndet_example = scripts.generate_example:main",
            "nndet_prep = scripts.preprocess:main",
            "nndet_prep_labels = scripts.preprocess:main_prep_labels",
            "nndet_cls2fg = scripts.convert_cls2fg:main",
            "nndet_seg2det = scripts.convert_seg2det:main",
            "nndet_train = scripts.train:train",
            "nndet_sweep = scripts.train:sweep",
            "nndet_pretrain = scripts.pretrain:pretrain",
            "nndet_eval = scripts.train:evaluate",
            "nndet_predict = scripts.predict:main",
            "nndet_consolidate = scripts.consolidate:main",
            "nndet_boxes2nii = scripts.utils:boxes2nii",
            "nndet_boxes2nii2 = scripts.utils:boxes2nii2",
            "nndet_seg2nii = scripts.utils:seg2nii",
            "nndet_unpack = scripts.utils:unpack",
            "nndet_env = scripts.utils:env",
            "nndet_print_reg = scripts.utils:print_reg",
            "nndet_test_split = scripts.utils:create_test_split",
        ]
    },
)
