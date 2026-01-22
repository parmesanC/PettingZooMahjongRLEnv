from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "mahjong_win_checker",
        ["mahjong_win_checker.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="mahjong_win_checker",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)


