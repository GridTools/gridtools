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

#include <gridtools/stencil-composition/bind_functor_with_interval.hpp>

#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/stencil-composition/interval.hpp>
#include <gridtools/stencil-composition/level.hpp>

namespace gridtools {
    namespace {

        template <uint_t Splitter, int_t Offset>
        using lev = level<Splitter, Offset, 3>;

        template <class BoundFunctor>
        char const *run_bound_functor() {
            char const *res;
            BoundFunctor::Do(res);
            return res;
        }

        template <>
        char const *run_bound_functor<void>() {
            return "not defined";
        }

        template <class Functor, uint_t Splitter, int_t Offset>
        using testee = GT_META_CALL(
            bind_functor_with_interval, (Functor, GT_META_CALL(level_to_index, (lev<Splitter, Offset>))));

        template <class Functor, uint_t Splitter, int_t Offset>
        const char *run() {
            return run_bound_functor<testee<Functor, Splitter, Offset>>();
        }

        template <uint_t Splitter, int_t Offset>
        GT_META_DEFINE_ALIAS(idx, level_to_index, (lev<Splitter, Offset>));

        struct simple_functor {
            using param_list = std::tuple<>;

            template <class Eval>
            static GT_FUNCTION void Do(Eval &eval) {
                eval = "simple";
            }
        };

        // bound functor that has no interval overloads is the same as original functor
        static_assert(std::is_same<testee<simple_functor, 0, 1>, simple_functor>{}, "");

        TEST(bind_functor_with_interval, simple) { EXPECT_EQ("simple", (run<simple_functor, 0, 1>())); }

        struct one_interval_functor {
            using param_list = std::tuple<>;

            template <class Eval>
            static GT_FUNCTION void Do(Eval &eval, interval<lev<0, 1>, lev<1, 1>>) {
                eval = "one interval";
            }
        };

        // bound functor is the same within validity area
        static_assert(std::is_same<testee<one_interval_functor, 0, 1>, testee<one_interval_functor, 0, 2>>{}, "");

        TEST(bind_functor_with_interval, one_interval) {
            EXPECT_EQ("not defined", (run<one_interval_functor, 0, -1>()));
            EXPECT_EQ("one interval", (run<one_interval_functor, 0, 1>()));
            EXPECT_EQ("one interval", (run<one_interval_functor, 0, 2>()));
            EXPECT_EQ("one interval", (run<one_interval_functor, 1, -1>()));
            EXPECT_EQ("one interval", (run<one_interval_functor, 1, 1>()));
            EXPECT_EQ("not defined", (run<one_interval_functor, 1, 2>()));
        }

        struct overloaded_functor {
            using param_list = std::tuple<>;

            template <class Eval>
            static GT_FUNCTION void Do(Eval &eval, interval<lev<0, 1>, lev<1, -1>>) {
                eval = "overload 1";
            }
            template <class Eval>
            static GT_FUNCTION void Do(Eval &eval, interval<lev<1, 1>, lev<2, -1>>) {
                eval = "overload 2";
            }
        };

        TEST(bind_functor_with_interval, overloaded) {
            EXPECT_EQ("not defined", (run<overloaded_functor, 0, -1>()));
            EXPECT_EQ("overload 1", (run<overloaded_functor, 0, 1>()));
            EXPECT_EQ("overload 1", (run<overloaded_functor, 0, 2>()));
            EXPECT_EQ("overload 1", (run<overloaded_functor, 1, -1>()));
            EXPECT_EQ("overload 2", (run<overloaded_functor, 1, 1>()));
            EXPECT_EQ("overload 2", (run<overloaded_functor, 1, 2>()));
            EXPECT_EQ("overload 2", (run<overloaded_functor, 2, -1>()));
            EXPECT_EQ("not defined", (run<overloaded_functor, 2, 1>()));
        }

        struct with_default_functor {
            using param_list = std::tuple<>;

            template <class Eval>
            static GT_FUNCTION void Do(Eval &eval) {
                eval = "default";
            }
            template <class Eval>
            static GT_FUNCTION void Do(Eval &eval, interval<lev<0, 1>, lev<1, 1>>) {
                eval = "interval";
            }
        };

        TEST(bind_functor_with_interval, with_default) {
            EXPECT_EQ("default", (run<with_default_functor, 0, -1>()));
            EXPECT_EQ("interval", (run<with_default_functor, 0, 1>()));
            EXPECT_EQ("interval", (run<with_default_functor, 0, 2>()));
            EXPECT_EQ("interval", (run<with_default_functor, 1, -1>()));
            EXPECT_EQ("interval", (run<with_default_functor, 1, 1>()));
            EXPECT_EQ("default", (run<with_default_functor, 1, 2>()));
        }

        struct int_functor {
            using param_list = std::tuple<>;
            template <class Eval>
            static GT_FUNCTION int Do(Eval &, interval<lev<0, 1>, lev<1, 1>>) {
                return 42;
            }
        };

        TEST(bind_functor_with_interval, return_value) {
            int dummy;
            EXPECT_EQ(42, (testee<int_functor, 0, 1>::Do(dummy)));
        }
    } // namespace
} // namespace gridtools
