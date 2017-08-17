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
#include <common/variadic_pack_metafunctions.hpp>
#include <common/defs.hpp>

TEST(VariadicPackMetafunctions, GetIndexOfElementInVariadicPack) {
    GRIDTOOLS_STATIC_ASSERT((gridtools::get_index_of_element_in_pack(0, 1, 1, 2, 3, 4) == 0),
        "Failed to retrieve correct index from varidadic pack.");
    GRIDTOOLS_STATIC_ASSERT((gridtools::get_index_of_element_in_pack(0, 2, 1, 2, 3, 4) == 1),
        "Failed to retrieve correct index from varidadic pack.");
    GRIDTOOLS_STATIC_ASSERT((gridtools::get_index_of_element_in_pack(0, 3, 1, 2, 3, 4) == 2),
        "Failed to retrieve correct index from varidadic pack.");
    GRIDTOOLS_STATIC_ASSERT((gridtools::get_index_of_element_in_pack(0, 4, 1, 2, 3, 4) == 3),
        "Failed to retrieve correct index from varidadic pack.");
}

TEST(VariadicPackMetafunctions, GetElementFromVariadicPack) {
    GRIDTOOLS_STATIC_ASSERT(
        (gridtools::get_value_from_pack(0, 1, 2, 3, 4) == 1), "Failed to retrieve correct value from varidadic pack.");
    GRIDTOOLS_STATIC_ASSERT(
        (gridtools::get_value_from_pack(1, 1, 2, 3, 4) == 2), "Failed to retrieve correct value from varidadic pack.");
    GRIDTOOLS_STATIC_ASSERT(
        (gridtools::get_value_from_pack(2, 1, 2, 3, 4) == 3), "Failed to retrieve correct value from varidadic pack.");
    GRIDTOOLS_STATIC_ASSERT(
        (gridtools::get_value_from_pack(3, 1, 2, 3, 4) == 4), "Failed to retrieve correct value from varidadic pack.");
}
