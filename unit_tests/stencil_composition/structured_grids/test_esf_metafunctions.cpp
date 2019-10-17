/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/compute_extents_metafunctions.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>

using namespace gridtools;

struct functor0 {
    typedef accessor<0, intent::in, extent<0, 0, -1, 3, -2, 0>> in0;
    typedef accessor<1, intent::in, extent<-1, 1, 0, 2, -1, 2>> in1;
    typedef accessor<2, intent::in, extent<-3, 3, -1, 2, 0, 1>> in2;
    typedef accessor<3, intent::inout> out;

    typedef make_param_list<in0, in1, in2, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor1 {
    typedef accessor<0, intent::in, extent<0, 1, -1, 2, 0, 0>> in0;
    typedef accessor<1, intent::inout> out;
    typedef accessor<2, intent::in, extent<-3, 0, -3, 0, 0, 2>> in2;
    typedef accessor<3, intent::in, extent<0, 2, 0, 2, -2, 3>> in3;

    typedef make_param_list<in0, out, in2, in3> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor2 {
    typedef accessor<0, intent::in, extent<-3, 3, -1, 0, -2, 1>> in0;
    typedef accessor<1, intent::in, extent<-3, 1, -2, 1, 0, 2>> in1;
    typedef accessor<2, intent::inout> out;

    typedef make_param_list<in0, in1, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor3 {
    typedef accessor<0, intent::in, extent<0, 3, 0, 1, -2, 0>> in0;
    typedef accessor<1, intent::in, extent<-2, 3, 0, 2, -3, 1>> in1;
    typedef accessor<2, intent::inout> out;
    typedef accessor<3, intent::in, extent<-1, 3, -3, 0, -3, 2>> in3;

    typedef make_param_list<in0, in1, out, in3> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor4 {
    typedef accessor<0, intent::in, extent<0, 3, -2, 1, -3, 2>> in0;
    typedef accessor<1, intent::in, extent<-2, 3, 0, 3, -3, 2>> in1;
    typedef accessor<2, intent::in, extent<-1, 1, 0, 3, 0, 3>> in2;
    typedef accessor<3, intent::inout> out;

    typedef make_param_list<in0, in1, in2, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor5 {
    typedef accessor<0, intent::in, extent<-3, 1, -1, 2, -1, 1>> in0;
    typedef accessor<1, intent::in, extent<0, 1, -2, 2, 0, 3>> in1;
    typedef accessor<2, intent::in, extent<0, 2, 0, 3, -1, 2>> in2;
    typedef accessor<3, intent::inout> out;

    typedef make_param_list<in0, in1, in2, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

struct functor6 {
    typedef accessor<0, intent::inout> out;
    typedef accessor<1, intent::in, extent<0, 3, -3, 2, 0, 0>> in1;
    typedef accessor<2, intent::in, extent<-3, 2, 0, 2, -1, 2>> in2;
    typedef accessor<3, intent::in, extent<-1, 0, -1, 0, -1, 3>> in3;

    typedef make_param_list<out, in1, in2, in3> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation);
};

typedef arg<0> o0;
typedef arg<1> o1;
typedef arg<2> o2;
typedef arg<3> o3;
typedef arg<4> o4;
typedef arg<5> o5;
typedef arg<6> o6;
typedef arg<7> in0;
typedef arg<8> in1;
typedef arg<9> in2;
typedef arg<10> in3;

using mss_t = decltype(make_multistage(execute::forward(),
    make_stage<functor0>(in0(), in1(), in2(), o0()),
    make_stage<functor1>(in3(), o1(), in0(), o0()),
    make_stage<functor2>(o0(), o1(), o2()),
    make_stage<functor3>(in1(), in2(), o3(), o2()),
    make_stage<functor4>(o0(), o1(), o3(), o4()),
    make_stage<functor5>(in3(), o4(), in0(), o5()),
    make_stage<functor6>(o6(), o5(), in1(), in2())));

template <class Arg, int_t... ExpectedExtentValues>
using testee = std::is_same<lookup_extent_map<get_extent_map_from_mss<mss_t>, Arg>, extent<ExpectedExtentValues...>>;

static_assert(testee<o0, -5, 11, -10, 10, -5, 13>::value, "");
static_assert(testee<o1, -5, 9, -10, 8, -3, 10>::value, "");
static_assert(testee<o2, -2, 8, -8, 7, -3, 8>::value, "");
static_assert(testee<o3, -1, 5, -5, 7, 0, 6>::value, "");
static_assert(testee<o4, 0, 4, -5, 4, 0, 3>::value, "");
static_assert(testee<o5, 0, 3, -3, 2, 0, 0>::value, "");
static_assert(testee<o6, 0, 0, 0, 0, 0, 0>::value, "");
static_assert(testee<in0, -8, 11, -13, 13, -7, 13>::value, "");
static_assert(testee<in1, -6, 12, -10, 12, -6, 15>::value, "");
static_assert(testee<in2, -8, 14, -11, 12, -5, 14>::value, "");
static_assert(testee<in3, -5, 10, -11, 10, -3, 10>::value, "");

TEST(dummy, dummy) {}
