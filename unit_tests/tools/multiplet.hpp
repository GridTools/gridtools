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

#pragma once

#include "common/defs.hpp"
#include "common/array.hpp"
#include "common/array_addons.hpp" // included to pretty print the array in gtest
#include "common/generic_metafunctions/is_all_integrals.hpp"

template < size_t N >
using multiplet = gridtools::array< size_t, N >;
template < typename... Ts, typename std::enable_if< gridtools::is_all_integral< Ts... >::value, int >::type = 0 >
auto GT_FUNCTION make_multiplet(Ts... ts) GT_AUTO_RETURN((gridtools::array< size_t, sizeof...(Ts) >{(size_t)ts...}));

// nvcc doesn't like the variadic form
gridtools::array< size_t, 1 > GT_FUNCTION make_multiplet(size_t t0) { return {t0}; }
gridtools::array< size_t, 2 > GT_FUNCTION make_multiplet(size_t t0, size_t t1) { return {t0, t1}; }
gridtools::array< size_t, 3 > GT_FUNCTION make_multiplet(size_t t0, size_t t1, size_t t2) { return {t0, t1, t2}; }
