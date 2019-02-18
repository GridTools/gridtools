/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "grid_traits_fwd.hpp"

#ifdef GT_STRUCTURED_GRIDS
#include "structured_grids/grid_traits.hpp"
#else
#include "icosahedral_grids/grid_traits.hpp"
#endif
