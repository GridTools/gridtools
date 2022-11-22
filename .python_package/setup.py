# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

from pkg_resources import parse_version
from setuptools import find_packages
from skbuild import setup

if __name__ == "__main__":
    version_file = pathlib.Path(__file__).absolute().parent.parent / "version.txt"
    setup(
        package_dir={"": "py_src"},
        packages=find_packages(where="py_src"),
        version=str(parse_version(version_file.read_text())),
        package_data={"gridtools": ["*.hpp", "*.h", "*.cmake"]},
        cmake_install_dir="py_src/gridtools/data",
        cmake_args=["-DBUILD_TESTING=OFF", "-DGT_INSTALL_EXAMPLES:BOOL=OFF"],
        cmake_source_dir="../",
        cmake_with_sdist=True,
    )
