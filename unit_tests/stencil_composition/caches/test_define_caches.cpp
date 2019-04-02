/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
   @file
   @brief File containing tests for the define_cache construct
*/

#include <gtest/gtest.h>

#include <boost/mpl/equal.hpp>

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/stencil_composition/caches/define_caches.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace execute;

TEST(define_caches, test_sequence_caches) {
    typedef storage_traits<backend_t>::storage_info_t<0, 3> storage_info_t;
    typedef storage_traits<backend_t>::data_store_t<float_type, storage_info_t> storage_t;

    typedef gridtools::arg<0, storage_t> arg0_t;
    typedef gridtools::arg<1, storage_t> arg1_t;
    typedef gridtools::arg<2, storage_t> arg2_t;

    typedef decltype(gridtools::define_caches(cache<cache_type::ij, cache_io_policy::fill>(arg0_t()),
        cache<cache_type::k, cache_io_policy::local>(arg2_t()))) cache_sequence_t;

    GT_STATIC_ASSERT((boost::mpl::equal<cache_sequence_t,
                         boost::mpl::vector2<detail::cache_impl<cache_type::ij, arg0_t, cache_io_policy::fill>,
                             detail::cache_impl<cache_type::k, arg2_t, cache_io_policy::local>>>::value),
        "Failed TEST");

    typedef decltype(
        gridtools::cache<cache_type::k, cache_io_policy::flush>(arg0_t(), arg1_t())) caches_ret_sequence_4_t;
    typedef decltype(
        gridtools::cache<cache_type::ij, cache_io_policy::fill>(arg0_t(), arg1_t(), arg2_t())) caches_ret_sequence_3_t;
    typedef decltype(gridtools::cache<cache_type::ij, cache_io_policy::fill>(arg0_t())) caches_ret_sequence_1_t;

    GT_STATIC_ASSERT((boost::mpl::equal<caches_ret_sequence_4_t,
                         boost::mpl::vector2<detail::cache_impl<cache_type::k, arg0_t, cache_io_policy::flush>,
                             detail::cache_impl<cache_type::k, arg1_t, cache_io_policy::flush>>>::value),
        "Failed TEST");

    GT_STATIC_ASSERT((boost::mpl::equal<caches_ret_sequence_3_t,
                         boost::mpl::vector3<detail::cache_impl<cache_type::ij, arg0_t, cache_io_policy::fill>,
                             detail::cache_impl<cache_type::ij, arg1_t, cache_io_policy::fill>,
                             detail::cache_impl<cache_type::ij, arg2_t, cache_io_policy::fill>>>::value),
        "Failed TEST");
    GT_STATIC_ASSERT(
        (boost::mpl::equal<caches_ret_sequence_1_t,
            boost::mpl::vector1<detail::cache_impl<cache_type::ij, arg0_t, cache_io_policy::fill>>>::value),
        "Failed TEST");
}
