/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;

struct stage1 {
    using in1 = in_accessor<0, extent<0, 1, -1, 0, 0, 1>>;
    using in2 = in_accessor<1, extent<0, 1, -1, 0, -1, 1>>;
    using out = inout_accessor<2>;
    using param_list = make_param_list<in1, in2, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&) {}
};

struct stage2 {
    using in1 = in_accessor<0, extent<-1, 0, 0, 1, -1, 0>>;
    using in2 = in_accessor<1, extent<-1, 1, -1, 0, -1, 1>>;
    using out = inout_accessor<2>;
    using param_list = make_param_list<in1, in2, out>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&) {}
};

template <class Plh, class Expected, class Mss>
constexpr bool test_extent(Mss) {
    return std::is_same<decltype(get_arg_extent<Mss>(Plh())), Expected>::value;
}

template <class Plh, class Mss>
constexpr bool test_intent(Mss, intent expected) {
    return decltype(get_arg_intent<Mss>(Plh()))::value == expected;
}

using p_in1 = arg<0>;
using p_in2 = arg<1>;
using p_tmp1 = arg<2>;
using p_tmp2 = arg<3>;
using p_tmp3 = arg<4>;
using p_out = arg<5>;

constexpr auto mss0 = make_multistage(execute::forward(), make_stage<stage1>(p_in1(), p_in2(), p_out()));

static_assert(test_extent<p_in1, extent<0, 1, -1, 0, 0, 1>>(mss0), "");
static_assert(test_extent<p_in2, extent<0, 1, -1, 0, -1, 1>>(mss0), "");
static_assert(test_extent<p_out, extent<>>(mss0), "");

static_assert(test_intent<p_in1>(mss0, intent::in), "");
static_assert(test_intent<p_in2>(mss0, intent::in), "");
static_assert(test_intent<p_out>(mss0, intent::inout), "");

constexpr auto mss1 = make_multistage(
    execute::forward(), make_stage<stage1>(p_in1(), p_in2(), p_tmp1()), make_stage<stage2>(p_in1(), p_tmp1(), p_out()));

static_assert(test_extent<p_in1, extent<-1, 2, -2, 1, -1, 2>>(mss1), "");
static_assert(test_extent<p_in2, extent<-1, 2, -2, 0, -2, 2>>(mss1), "");
static_assert(test_extent<p_tmp1, extent<-1, 1, -1, 0, -1, 1>>(mss1), "");
static_assert(test_extent<p_out, extent<>>(mss1), "");

static_assert(test_intent<p_in1>(mss1, intent::in), "");
static_assert(test_intent<p_in2>(mss1, intent::in), "");
static_assert(test_intent<p_tmp1>(mss1, intent::inout), "");
static_assert(test_intent<p_out>(mss1, intent::inout), "");

constexpr auto mss2 = make_multistage(execute::forward(),
    make_stage<stage1>(p_in1(), p_in2(), p_tmp1()),
    make_stage<stage1>(p_in1(), p_tmp1(), p_tmp2()),
    make_stage<stage2>(p_in2(), p_tmp1(), p_tmp3()),
    make_stage<stage2>(p_tmp2(), p_tmp3(), p_out()));

// after last stage:
//   p_out:  {0, 0, 0, 0, 0, 0}
//   p_tmp3: {-1, 1, -1, 0, -1, 1}
//   p_tmp2: {-1, 0, 0, 1, -1, 0}
//
// after second independent stage:
//   p_out:  {0, 0, 0, 0, 0, 0}
//   p_tmp3: {-1, 1, -1, 0, -1, 1}
//   p_tmp2: {-1, 0, 0, 1, -1, 0}
//   p_tmp1: {-2, 2, -2, 0, -2, 2}
//   p_in2:  {-2, 1, -1, 1, -2, 1}
//
// after first independent stage:
//   p_out:  {0, 0, 0, 0, 0, 0}
//   p_tmp3: {-1, 1, -1, 0, -1, 1}
//   p_tmp2: {-1, 0, 0, 1, -1, 0}
//   p_tmp1: {-2, 2, -2, 1, -2, 2}
//   p_in2:  {-2, 1, -1, 1, -2, 1}
//   p_in1:  {-1, 1, -1, 1, -1, 1}
//
// after first stage
//   p_out:  {0, 0, 0, 0, 0, 0}
//   p_tmp3: {-1, 1, -1, 0, -1, 1}
//   p_tmp2: {-1, 0, 0, 1, -1, 0}
//   p_tmp1: {-2, 2, -2, 1, -2, 2}
//   p_in2:  {-2, 3, -3, 1, -3, 3}
//   p_in1:  {-2, 3, -3, 1, -2, 3}

static_assert(test_extent<p_in1, extent<-2, 3, -3, 1, -2, 3>>(mss2), "");
static_assert(test_extent<p_in2, extent<-2, 3, -3, 1, -3, 3>>(mss2), "");
static_assert(test_extent<p_tmp1, extent<-2, 2, -2, 1, -2, 2>>(mss2), "");
static_assert(test_extent<p_tmp2, extent<-1, 0, 0, 1, -1, 0>>(mss2), "");
static_assert(test_extent<p_tmp3, extent<-1, 1, -1, 0, -1, 1>>(mss2), "");
static_assert(test_extent<p_out, extent<>>(mss2), "");

static_assert(test_intent<p_in1>(mss2, intent::in), "");
static_assert(test_intent<p_in2>(mss2, intent::in), "");
static_assert(test_intent<p_tmp1>(mss2, intent::inout), "");
static_assert(test_intent<p_tmp2>(mss2, intent::inout), "");
static_assert(test_intent<p_tmp3>(mss2, intent::inout), "");
static_assert(test_intent<p_out>(mss2, intent::inout), "");

TEST(dummy, dummy) {}
