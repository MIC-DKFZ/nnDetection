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
                requirements += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
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
    source_cpu = []  # list((extensions_dir/'cpu').glob('*.cpp')) temporary until I added header files ...
    source_cuda = list((extensions_dir / "cuda").glob("*.cu"))
    print("main_file {}".format(main_file))
    print("source_cpu {}".format(source_cpu))
    print("source_cuda {}".format(source_cuda))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []
    extra_compile_args = {"cxx": []}

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
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
    # package_data={"nndet": {"*.so"}},
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
            # preprocessing + preparation
            "nndet_prep = nndet_scripts.preprocess:main",
            "nndet_prep_labels = nndet_scripts.preprocess:main_prep_labels",
            "nndet_cv_split = nndet_scripts.utils:create_cv_split",
            # training
            "nndet_train = nndet_scripts.train:train",
            "nndet_sweep = nndet_scripts.train:sweep",
            "nndet_consolidate = nndet_scripts.consolidate:main",
            # evaluation
            "nndet_eval = nndet_scripts.train:evaluate",
            "nndet_eval_with_folders = nndet_scripts.train:evaluate_with_folders",
            # prediction
            "nndet_predict_with_task = nndet_scripts.predict2:entrypoint_predict_with_task",
            "nndet_predict_with_folders = nndet_scripts.predict2:entrypoint_predict_with_folders",
            "nndet_predict_test_split = nndet_scripts.predict2:entrypoint_predict_test_split",
            "nndet_preprocess_for_inference = nndet_scripts.predict2:entrypoint_preprocess_for_inference",
            # manual ensembling
            "nndet_ensemble_with_task = nndet_scripts.ensemble:entrypoint_ensemble_with_task",
            "nndet_ensemble_with_models = nndet_scripts.ensemble:entrypoint_ensemble_with_models",
            "nndet_ensemble_with_folders = nndet_scripts.ensemble:entrypoint_ensemble_with_folders",
            # utils
            "nndet_predict = nndet_scripts.predict:main",  # deprecated
            "nndet_example = nndet_scripts.generate_example:main",
            "nndet_cls2fg = nndet_scripts.convert_cls2fg:main",
            "nndet_seg2det = nndet_scripts.convert_seg2det:main",
            "nndet_pretrain = nndet_scripts.pretrain:pretrain",
            "nndet_boxes2nii = nndet_scripts.utils:boxes2nii",
            "nndet_boxes2nii2 = nndet_scripts.utils:boxes2nii2",
            "nndet_masks2nii = nndet_scripts.utils:masks2nii",
            "nndet_seg2nii = nndet_scripts.utils:seg2nii",
            "nndet_unpack = nndet_scripts.utils:unpack",
            "nndet_unpack_task = nndet_scripts.utils:unpack_task",
            "nndet_env = nndet_scripts.utils:env",
            "nndet_print_reg = nndet_scripts.utils:print_reg",
            "nndet_test_data_split = nndet_scripts.utils:create_test_data_split",
        ]
    },
)
