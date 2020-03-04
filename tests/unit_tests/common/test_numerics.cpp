/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/numerics.hpp>

template <unsigned N>
constexpr auto testee = gridtools::_impl::static_pow3<N>::value;

static_assert(testee<0> == 1, "");
static_assert(testee<1> == 3, "");
static_assert(testee<2> == 9, "");
static_assert(testee<3> == 27, "");
static_assert(testee<4> == 81, "");
