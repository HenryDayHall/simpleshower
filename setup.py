from setuptools import setup

# subshowers (https://github.com/HenryDayHall/subshowers) is needed too; it
# isn't on PyPI, so install it from its repository, or put it on PYTHONPATH.
setup(
    name="simpleshower",
    version="0.1.0",
    packages=["simpleshower"],
    install_requires=["numpy", "pyyaml", "torch"],
    # pip install -e .[corsika]: exporting bundles, and sampling them offline
    extras_require={"corsika": ["onnx", "onnxscript", "onnxruntime", "pyarrow"]},
)
