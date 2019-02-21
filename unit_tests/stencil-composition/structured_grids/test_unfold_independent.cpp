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

#include <gridtools/stencil-composition/esf_metafunctions.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>

namespace gridtools {
    namespace {
        struct functor {
            using a0 = accessor<0, intent::inout>;
            using a1 = accessor<1, intent::inout>;

            using param_list = make_param_list<a0, a1>;
        };

        struct fake_storage_type {
            using value_type = int;
        };

        constexpr arg<0, fake_storage_type> p0 = {};
        constexpr arg<1, fake_storage_type> p1 = {};
        constexpr auto stage = make_stage<functor>(p0, p1);

        using mss_type = decltype(make_multistage(
            execute::forward(), stage, stage, stage, make_independent(stage, stage, make_independent(stage, stage))));

        using testee_t = GT_META_CALL(unwrap_independent, mss_type::esf_sequence_t);

        static_assert(meta::length<testee_t>::value == 7, "");
        static_assert(meta::all_of<is_esf_descriptor, testee_t>::value, "");

        TEST(dummy, dumy) {}
    } // namespace
} // namespace gridtools
