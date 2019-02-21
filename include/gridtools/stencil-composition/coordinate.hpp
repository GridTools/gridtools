/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

/**
 *  @file
 *
 *  Specialisations of coord_i<Backend>, coord_j<Backend>, coord_k<Backend> are defined here.
 *  Backend should be an instantiation of backend_ids.
 *  coord_* are expected to be like std::integral_constant of size_t
 */

#ifdef GT_STRUCTURED_GRIDS
#include "./structured_grids/coordinate.hpp"
#else
#include "./icosahedral_grids/coordinate.hpp"
#endif
