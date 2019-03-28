/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/caches/cache_metafunctions.hpp>
#include <gridtools/stencil_composition/caches/extract_extent_caches.hpp>

#include <boost/fusion/include/pair.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace execute;

typedef storage_traits<backend::x86>::storage_info_t<0, 2> storage_info_ij_t;
typedef storage_traits<backend::x86>::data_store_t<float_type, storage_info_ij_t> storage_type;

typedef arg<0, storage_type> p_in;
typedef arg<2, storage_type> p_out;
typedef arg<1, storage_type> p_buff;
typedef arg<3, storage_type> p_notin;

struct functor2 {
    typedef accessor<0, intent::in, extent<0, 0, 0, 0, -1, 0>> in;
    typedef accessor<1, intent::inout, extent<0, 0, 0, 0, 0, 1>> out;
    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval);
};

typedef detail::cache_impl<cache_type::ij, p_in, cache_io_policy::fill> cache1_t;
typedef detail::cache_impl<cache_type::ij, p_buff, cache_io_policy::fill> cache2_t;
typedef detail::cache_impl<cache_type::k, p_out, cache_io_policy::local> cache3_t;
typedef detail::cache_impl<cache_type::k, p_notin, cache_io_policy::local> cache4_t;
typedef std::tuple<cache1_t, cache2_t, cache3_t, cache4_t> caches_t;

TEST(cache_metafunctions, get_ij_cache_storage_tuple) {
    using testee_t = get_ij_cache_storage_tuple<caches_t, extent<-2, 2, -3, 2>, 32, 4>::type;

    using expected_t = std::tuple<boost::fusion::pair<p_in, ij_cache_storage<float_type, 36, 9, 2, 3>>,
        boost::fusion::pair<p_buff, ij_cache_storage<float_type, 36, 9, 2, 3>>>;

    static_assert(std::is_same<testee_t, expected_t>::value, "");
}

using esf1k_t = decltype(make_stage<functor2>(p_in(), p_notin()));
using esf2k_t = decltype(make_stage<functor2>(p_notin(), p_out()));

using esfk_sequence_t = meta::list<esf1k_t, esf2k_t>;

static_assert(
    std::is_same<GT_META_CALL(extract_k_extent_for_cache, (p_out, esfk_sequence_t)), extent<0, 0, 0, 0, 0, 1>>(), "");

static_assert(
    std::is_same<GT_META_CALL(extract_k_extent_for_cache, (p_notin, esfk_sequence_t)), extent<0, 0, 0, 0, -1, 1>>(),
    "");

TEST(cache_metafunctions, get_k_cache_storage_tuple) {

    using testee_t = typename get_k_cache_storage_tuple<caches_t, esfk_sequence_t>::type;

    using expected_t = std::tuple<boost::fusion::pair<p_out, k_cache_storage<p_out, float_type, 0, 1>>,
        boost::fusion::pair<p_notin, k_cache_storage<p_notin, float_type, -1, 1>>>;

    static_assert(std::is_same<testee_t, expected_t>::value, "");
}
