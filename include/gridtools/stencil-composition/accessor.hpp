/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#ifdef GT_STRUCTURED_GRIDS
#include "./structured_grids/accessor.hpp"
#include "./structured_grids/accessor_mixed.hpp"
#else
#include "./icosahedral_grids/accessor.hpp"
#endif

#include "./global_accessor.hpp"
