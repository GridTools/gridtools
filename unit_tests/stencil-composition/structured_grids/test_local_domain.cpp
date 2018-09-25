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
#include <gridtools/common/generic_metafunctions/meta.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/layout_map.hpp>
#include <gridtools/stencil-composition/accessor.hpp>
#include <gridtools/stencil-composition/arg.hpp>
#include <gridtools/stencil-composition/axis.hpp>
#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/grid.hpp>
#include <gridtools/stencil-composition/intermediate.hpp>
#include <gridtools/stencil-composition/make_stage.hpp>
#include <gridtools/stencil-composition/make_stencils.hpp>
#include <gridtools/storage/storage-facility.hpp>
#include <gridtools/storage/storage_host/host_storage_info.hpp>

using namespace gridtools;
using namespace enumtype;

// These are the stencil operators that compose the multistage stencil in this test
struct dummy_functor {
    typedef accessor<0, inout> in;
    typedef accessor<1> out;
    typedef boost::mpl::vector<in, out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval);
};

typedef backend<platform::x86, GRIDBACKEND, strategy::naive> backend_t;
typedef layout_map<2, 1, 0> layout_ijk_t;
typedef layout_map<0, 1, 2> layout_kji_t;
typedef host_storage_info<0, layout_ijk_t> meta_ijk_t;
typedef host_storage_info<0, layout_kji_t> meta_kji_t;
typedef storage_traits<backend_t::backend_id_t>::data_store_t<float_type, meta_ijk_t> storage_t;
typedef storage_traits<backend_t::backend_id_t>::data_store_t<float_type, meta_kji_t> storage_buff_t;

typedef arg<0, storage_t> p_in;
typedef arg<1, storage_buff_t> p_buff;
typedef arg<2, storage_t> p_out;

typedef intermediate<false,
    backend<platform::x86, GRIDBACKEND, strategy::naive>,
    grid<axis<1>::axis_interval_t>,
    std::tuple<>,
    std::tuple<decltype(make_multistage // mss_descriptor
        (execute<forward>(),
            make_stage<dummy_functor>(p_in(), p_buff()),
            make_stage<dummy_functor>(p_buff(), p_out())))>>
    intermediate_t;

using local_domains_t = intermediate_local_domains<intermediate_t>;

static_assert(meta::length<local_domains_t>{} == 2, "");

using local_domain1_t = GT_META_CALL(meta::first, local_domains_t);

// local domain should contain the args used by all the esfs
static_assert(std::is_same<typename local_domain1_t::esf_args, std::tuple<p_in, p_buff>>{}, "");

using local_domain2_t = GT_META_CALL(meta::second, local_domains_t);

// local domain should contain the args used by all the esfs
static_assert(std::is_same<typename local_domain2_t::esf_args, std::tuple<p_buff, p_out>>{}, "");

// icc build fails to build unit tests without a single test.
TEST(dummy, dummy) {}
