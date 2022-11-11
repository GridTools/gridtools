# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gridtools import get_cmake_dir


def test_cmake_dir_contains_gridtools_cmake():
    main_config_file = get_cmake_dir() / "GridToolsConfig.cmake"
    assert main_config_file.exists()
    assert main_config_file.read_text()
