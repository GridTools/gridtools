/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/bind_functor_with_interval.hpp>

#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/stencil_composition/interval.hpp>
#include <gridtools/stencil_composition/level.hpp>

namespace gridtools {
    namespace {

        template <uint_t Splitter, int_t Offset>
        using lev = level<Splitter, Offset, 3>;

        template <class BoundFunctor>
        char const *run_bound_functor() {
            char const *res;
            BoundFunctor::apply(res);
            return res;
        }

        template <>
        char const *run_bound_functor<void>() {
            return "not defined";
        }

        template <class Functor, uint_t Splitter, int_t Offset>
        using testee = bind_functor_with_interval<Functor, level_to_index<lev<Splitter, Offset>>>;

        template <class Functor, uint_t Splitter, int_t Offset>
        const char *run() {
            return run_bound_functor<testee<Functor, Splitter, Offset>>();
        }

        template <uint_t Splitter, int_t Offset>
        using idx = level_to_index<lev<Splitter, Offset>>;

        struct simple_functor {
            using param_list = std::tuple<>;

            template <class Eval>
            static GT_FUNCTION void apply(Eval &eval) {
                eval = "simple";
            }
        };

        // bound functor that has no interval overloads is the same as original functor
        static_assert(std::is_same<testee<simple_functor, 0, 1>, simple_functor>{}, "");

        TEST(bind_functor_with_interval, simple) { EXPECT_EQ("simple", (run<simple_functor, 0, 1>())); }

        struct one_interval_functor {
            using param_list = std::tuple<>;

            template <class Eval>
            static GT_FUNCTION void apply(Eval &eval, interval<lev<0, 1>, lev<1, 1>>) {
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
            static GT_FUNCTION void apply(Eval &eval, interval<lev<0, 1>, lev<1, -1>>) {
                eval = "overload 1";
            }
            template <class Eval>
            static GT_FUNCTION void apply(Eval &eval, interval<lev<1, 1>, lev<2, -1>>) {
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
            static GT_FUNCTION void apply(Eval &eval) {
                eval = "default";
            }
            template <class Eval>
            static GT_FUNCTION void apply(Eval &eval, interval<lev<0, 1>, lev<1, 1>>) {
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
            static GT_FUNCTION int apply(Eval &, interval<lev<0, 1>, lev<1, 1>>) {
                return 42;
            }
        };

        TEST(bind_functor_with_interval, return_value) {
            int dummy;
            EXPECT_EQ(42, (testee<int_functor, 0, 1>::apply(dummy)));
        }
    } // namespace
} // namespace gridtools
