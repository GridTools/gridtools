# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import pathlib

from pkg_resources import parse_version
from setuptools import find_packages
from skbuild import setup

if __name__ == "__main__":
    version_file = pathlib.Path(__file__).absolute().parent.parent / "version.txt"
    setup(
        package_dir={"": "py_src"},
        packages=find_packages(where="py_src"),
        version=parse_version(version_file.read_text()),
        cmake_install_dir="py_src/gridtools/data",
        cmake_args=["-DBUILD_TESTING=OFF", "-DGT_INSTALL_EXAMPLES:BOOL=OFF"],
        cmake_source_dir="../",
    )
