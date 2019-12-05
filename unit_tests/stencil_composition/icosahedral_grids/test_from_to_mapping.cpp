/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"

#include <gridtools/stencil_composition/icosahedral.hpp>

using namespace gridtools;
using namespace icosahedral;

// The purpose of this set of tests is to guarantee that the offsets methods of the different specializations
// provided by the connectivity tables in from<>::to<>::with_color return a constexpr array
// It is not intended to check here the actual value of the offsets, this would only replicate the values coded
// in the tables

template <class From, class To, uint_t Color>
constexpr bool testee = from<From>::template to<To>::template with_color<Color>::offsets()[0][0] < 10;

// From Cells to XXX
static_assert(testee<cells, cells, 0>, "");
static_assert(testee<cells, cells, 1>, "");
static_assert(testee<cells, edges, 0>, "");
static_assert(testee<cells, edges, 1>, "");
static_assert(testee<cells, vertices, 0>, "");
static_assert(testee<cells, vertices, 1>, "");

// From Edges to XXX
static_assert(testee<edges, cells, 0>, "");
static_assert(testee<edges, cells, 1>, "");
static_assert(testee<edges, cells, 2>, "");
static_assert(testee<edges, edges, 0>, "");
static_assert(testee<edges, edges, 1>, "");
static_assert(testee<edges, edges, 2>, "");
static_assert(testee<edges, vertices, 0>, "");
static_assert(testee<edges, vertices, 1>, "");
static_assert(testee<edges, vertices, 2>, "");

// From Vertexes to XXX
static_assert(testee<vertices, cells, 0>, "");
static_assert(testee<vertices, edges, 0>, "");
static_assert(testee<vertices, vertices, 0>, "");

TEST(dummy, dummy) {}
