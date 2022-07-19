import os

from setuptools import find_packages, setup


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


_SRC = os.path.dirname(__file__)
_REQ = os.path.join(_SRC, "requirements")

base_req = resolve_requirements(os.path.join(_REQ, "base.txt"))
extras = {
    "dev": resolve_requirements(os.path.join(_REQ, "dev.txt")),
}
readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))

setup(
    name="nndet_[project]",
    version="0.0.1",
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=base_req,
    python_requires=">=3.8",
    author="Division of Medical Image Computing, German Cancer Research Center",
    maintainer_email="m.baumgartner@dkfz-heidelberg.de",
    extras_require=extras,
)
