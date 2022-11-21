# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib


_file = pathlib.Path(__file__)


def get_cmake_dir() -> pathlib.Path:
    return _file.parent / "data" / "lib" / "cmake" / "GridTools"


def get_include_dir() -> pathlib.Path:
    return _file.parent / "data" / "include"
