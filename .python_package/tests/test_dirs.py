# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gridtools


def test_cmake_dir_contains_gridtools_cmake():
    main_config_file = gridtools.get_cmake_dir() / "GridToolsConfig.cmake"
    assert main_config_file.exists()
    assert main_config_file.read_text()


def test_include_dir_contains_headers():
    include_path = gridtools.get_include_dir()
    assert include_path.exists()
    assert len(list(include_path.rglob("*.hpp"))) > 0
