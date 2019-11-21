/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// this file will be preprocessed to visualize a repository
#include <gridtools/interface/repository.hpp>
GT_DEFINE_REPOSITORY(my_repository, (ijk, ijk_builder)(ij, ij_builder), (ijk, u)(ijk, v)(ij, crlat));
