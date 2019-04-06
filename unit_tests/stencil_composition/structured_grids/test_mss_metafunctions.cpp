/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <boost/mpl/equal.hpp>
#include <boost/mpl/vector.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace execute;

struct functor1 {
    GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(in), GT_INOUT_ACCESSOR(out));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
typedef storage_traits<backend_t>::data_store_t<float_type, meta_data_t> storage_t;

typedef arg<0, storage_t> p_in;
typedef arg<1, storage_t> p_out;
typedef tmp_arg<2, storage_t> p_buff;

TEST(mss_metafunctions, extract_mss_caches_and_esfs) {
    meta_data_t meta_(10, 10, 10);
    storage_t in(meta_, 1.0), out(meta_, 1.0);

    typedef decltype(make_stage<functor1>(p_in(), p_buff())) esf1_t;
    typedef decltype(make_stage<functor1>(p_buff(), p_out())) esf2_t;

    typedef decltype(make_multistage(execute::forward(),
        define_caches(cache<cache_type::ij, cache_io_policy::local>(p_buff(), p_out())),
        esf1_t(), // esf_descriptor
        esf2_t()  // esf_descriptor
        )) mss_t;
    GT_STATIC_ASSERT((boost::mpl::equal<mss_t::esf_sequence_t, boost::mpl::vector<esf1_t, esf2_t>>::value), "ERROR");

#ifndef GT_DISABLE_CACHING
    GT_STATIC_ASSERT((boost::mpl::equal<mss_t::cache_sequence_t,
                         boost::mpl::vector2<detail::cache_impl<cache_type::ij, p_buff, cache_io_policy::local>,
                             detail::cache_impl<cache_type::ij, p_out, cache_io_policy::local>>>::value),
        "ERROR\nLists do not match");
#else
    GT_STATIC_ASSERT((boost::mpl::empty<mss_t::cache_sequence_t>::value), "ERROR\nList not empty");
#endif

    ASSERT_TRUE(true);
}
