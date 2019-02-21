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

#ifdef GT_STRUCTURED_GRIDS
#include "./structured_grids/backend.hpp"
#else
#include "./icosahedral_grids/backend.hpp"
#endif
