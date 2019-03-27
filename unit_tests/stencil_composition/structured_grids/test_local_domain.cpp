/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * test_local_domain.cpp
 *
 *  Created on: Apr 9, 2015
 *      Author: carlosos
 */
#include <tuple>
#include <type_traits>

#include <boost/mpl/vector.hpp>
#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/layout_map.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/stencil_composition/accessor.hpp>
#include <gridtools/stencil_composition/arg.hpp>
#include <gridtools/stencil_composition/axis.hpp>
#include <gridtools/stencil_composition/backend.hpp>
#include <gridtools/stencil_composition/grid.hpp>
#include <gridtools/stencil_composition/intermediate.hpp>
#include <gridtools/stencil_composition/make_stage.hpp>
#include <gridtools/stencil_composition/make_stencils.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace execute;

// These are the stencil operators that compose the multistage stencil in this test
struct dummy_functor {
    typedef accessor<0, intent::inout> in;
    typedef accessor<1> out;
    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval);
};

typedef backend<target::naive> naive_backend_t;
typedef layout_map<2, 1, 0> layout_ijk_t;
typedef layout_map<0, 1, 2> layout_kji_t;
typedef storage_info<0, layout_ijk_t> meta_ijk_t;
typedef storage_info<0, layout_kji_t> meta_kji_t;
typedef storage_traits<target::naive>::data_store_t<float_type, meta_ijk_t> storage_t;
typedef storage_traits<target::naive>::data_store_t<float_type, meta_kji_t> storage_buff_t;

typedef arg<0, storage_t> p_in;
typedef arg<1, storage_buff_t> p_buff;
typedef arg<2, storage_t> p_out;

typedef intermediate<false,
    backend<target::naive>,
    grid<axis<1>::axis_interval_t>,
    std::tuple<>,
    std::tuple<decltype(make_multistage // mss_descriptor
        (execute::forward(),
            make_stage<dummy_functor>(p_in(), p_buff()),
            make_stage<dummy_functor>(p_buff(), p_out())))>>
    intermediate_t;

using local_domains_t = intermediate_local_domains<intermediate_t>;

static_assert(meta::length<local_domains_t>{} == 2, "");

using local_domain1_t = GT_META_CALL(meta::first, local_domains_t);

// local domain should contain the args used by all the esfs
static_assert(std::is_same<typename local_domain1_t::esf_args_t, std::tuple<p_in, p_buff>>{}, "");

using local_domain2_t = GT_META_CALL(meta::second, local_domains_t);

// local domain should contain the args used by all the esfs
static_assert(std::is_same<typename local_domain2_t::esf_args_t, std::tuple<p_buff, p_out>>{}, "");

// icc build fails to build unit tests without a single test.
TEST(dummy, dummy) {}
