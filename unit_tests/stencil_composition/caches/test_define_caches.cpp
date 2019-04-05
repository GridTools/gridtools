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

#include <gridtools/stencil_composition/caches/define_caches.hpp>

#include <gtest/gtest.h>

#include <type_traits>

#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;

typedef storage_traits<backend_t>::storage_info_t<0, 3> storage_info_t;
typedef storage_traits<backend_t>::data_store_t<float_type, storage_info_t> storage_t;

typedef arg<0, storage_t> arg0_t;
typedef arg<1, storage_t> arg1_t;
typedef arg<2, storage_t> arg2_t;

typedef decltype(define_caches(cache<cache_type::ij, cache_io_policy::fill>(arg0_t()),
    cache<cache_type::k, cache_io_policy::local>(arg2_t()))) cache_sequence_t;

static_assert(std::is_same<cache_sequence_t,
                  std::tuple<detail::cache_impl<cache_type::ij, arg0_t, cache_io_policy::fill>,
                      detail::cache_impl<cache_type::k, arg2_t, cache_io_policy::local>>>::value,
    "");

typedef decltype(cache<cache_type::k, cache_io_policy::flush>(arg0_t(), arg1_t())) caches_ret_sequence_4_t;
typedef decltype(cache<cache_type::ij, cache_io_policy::fill>(arg0_t(), arg1_t(), arg2_t())) caches_ret_sequence_3_t;
typedef decltype(cache<cache_type::ij, cache_io_policy::fill>(arg0_t())) caches_ret_sequence_1_t;

static_assert(std::is_same<caches_ret_sequence_4_t,
                  std::tuple<detail::cache_impl<cache_type::k, arg0_t, cache_io_policy::flush>,
                      detail::cache_impl<cache_type::k, arg1_t, cache_io_policy::flush>>>::value,
    "");

static_assert(std::is_same<caches_ret_sequence_3_t,
                  std::tuple<detail::cache_impl<cache_type::ij, arg0_t, cache_io_policy::fill>,
                      detail::cache_impl<cache_type::ij, arg1_t, cache_io_policy::fill>,
                      detail::cache_impl<cache_type::ij, arg2_t, cache_io_policy::fill>>>::value,
    "");
static_assert(std::is_same<caches_ret_sequence_1_t,
                  std::tuple<detail::cache_impl<cache_type::ij, arg0_t, cache_io_policy::fill>>>::value,
    "");

TEST(dummy, dummy) {}
