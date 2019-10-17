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

#include <gridtools/stencil_composition/stencil_composition.hpp>

using namespace gridtools;

struct functor1 {
    typedef accessor<0> in;
    typedef accessor<1> out;
    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

typedef arg<0> p_in;
typedef arg<1> p_out;
typedef tmp_arg<2, float> p_buff;

typedef decltype(make_stage<functor1>(p_in(), p_buff())) esfs1_t;
typedef decltype(make_stage<functor1>(p_buff(), p_out())) esfs2_t;

typedef decltype(make_multistage(
    execute::forward(), define_caches(cache<cache_type::ij>(p_buff(), p_out())), esfs1_t(), esfs2_t())) mss_t;
static_assert(std::is_same<mss_t::esf_sequence_t, meta::list<esfs1_t, esfs2_t>>::value, "");

#ifndef GT_DISABLE_CACHING
static_assert(std::is_same<typename mss_t::cache_map_t,
                  meta::list<cache_info<p_buff, meta::list<cache_type::ij>, meta::list<>>,
                      cache_info<p_out, meta::list<cache_type::ij>, meta::list<>>>>::value,
    "");
#else
static_assert(std::is_same<mss_t::cache_map_t, meta::list<>>::value, "");
#endif

TEST(dummy, dummy) {}
