/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include "gtest/gtest.h"
#include <common/defs.hpp>
#include <common/gt_assert.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/icosahedral_grids/icosahedral_topology.hpp>

using namespace gridtools;

#ifdef __CUDACC__
#define BACKEND backend< gridtools::enumtype::Cuda, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< gridtools::enumtype::Host, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Block >
#else
#define BACKEND backend< gridtools::enumtype::Host, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Naive >
#endif
#endif

TEST(bakend, select_layout) {
#ifdef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< BACKEND::select_layout< selector< 1, 1, 1, 1 > >::type, layout_map< 3, 2, 1, 0 > >::value),
        "ERROR");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< BACKEND::select_layout< selector< 1, 0, 1, 1 > >::type, layout_map< 2, -1, 1, 0 > >::value),
        "ERROR");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< BACKEND::select_layout< selector< 1, 1, 0, 1, 1 > >::type,
                                layout_map< 3, 2, -1, 1, 0 > >::value),
        "ERROR");

#else
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< BACKEND::select_layout< selector< 1, 1, 1, 1 > >::type, layout_map< 0, 1, 2, 3 > >::value),
        "ERROR");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< BACKEND::select_layout< selector< 1, 0, 1, 1 > >::type, layout_map< 0, -1, 1, 2 > >::value),
        "ERROR");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< BACKEND::select_layout< selector< 1, 1, 0, 1, 1 > >::type,
                                layout_map< 1, 2, -1, 3, 0 > >::value),
        "ERROR");
#endif
}
