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

#include <gridtools/common/array.hpp>
#include <gridtools/common/array_addons.hpp>

template <gridtools::uint_t N>
using multiplet = gridtools::array<int, N>;

using triplet = multiplet<3>;
