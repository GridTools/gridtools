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

#include <gridtools/stencil-composition/make_loop_intervals.hpp>

#include <tuple>

#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/std_tuple.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/generic_metafunctions/meta.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/stencil-composition/bind_functor_with_interval.hpp>
#include <gridtools/stencil-composition/esf.hpp>
#include <gridtools/stencil-composition/interval.hpp>
#include <gridtools/stencil-composition/level.hpp>

namespace gridtools {
    namespace {
        using meta::list;

        template <uint_t Splitter, int_t Offset>
        using lev = level<Splitter, Offset, 3>;

        using from_t = lev<0, 1>;
        using to_t = lev<3, -1>;
        using axis_interval_t = interval<from_t, to_t>;

        template <class... Esfs>
        GT_META_DEFINE_ALIAS(testee, make_loop_intervals, (meta::list<Esfs...>, axis_interval_t));

        static_assert(meta::length<GT_META_CALL(testee, )>::value == 0, "");

        namespace simple {
            struct functor {
                using arg_list = std::tuple<>;

                template <class Eval>
                static GT_FUNCTION void Do(Eval &) {}
            };
            using esf_t = esf_descriptor<functor, std::tuple<>>;
            using testee_t = GT_META_CALL(testee, esf_t);
            static_assert(std::is_same<testee_t, list<loop_interval<from_t, to_t, list<esf_t>>>>{}, "");
        } // namespace simple

        namespace overlap {
            struct functor1 {
                using arg_list = std::tuple<>;

                template <class Eval>
                static GT_FUNCTION void Do(Eval &, interval<lev<0, 2>, lev<2, -1>>) {}
            };
            struct functor2 {
                using arg_list = std::tuple<>;

                template <class Eval>
                static GT_FUNCTION void Do(Eval &, interval<lev<1, 1>, lev<3, -2>>) {}
            };
            using esf1_t = esf_descriptor<functor1, std::tuple<>>;
            using esf2_t = esf_descriptor<functor2, std::tuple<>>;

            using bound_functor1 =
                bind_functor_with_interval<functor1, GT_META_CALL(level_to_index, (lev<0, 2>))>::type;
            using bound_functor2 =
                bind_functor_with_interval<functor2, GT_META_CALL(level_to_index, (lev<1, 1>))>::type;

            using bound_esf1 = esf_descriptor<bound_functor1, std::tuple<>>;
            using bound_esf2 = esf_descriptor<bound_functor2, std::tuple<>>;

            using testee_t = GT_META_CALL(testee, (esf1_t, esf2_t));

            using expected_t = list<loop_interval<lev<0, 2>, lev<1, -1>, list<bound_esf1>>,
                loop_interval<lev<1, 1>, lev<2, -1>, list<bound_esf1, bound_esf2>>,
                loop_interval<lev<2, 1>, lev<3, -2>, list<bound_esf2>>>;

            static_assert(std::is_same<testee_t, expected_t>{}, "");
        } // namespace overlap

        TEST(dummy, dummy) {}
    } // namespace
} // namespace gridtools
