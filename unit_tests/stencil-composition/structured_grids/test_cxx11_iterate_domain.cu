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
#define PEDANTIC_DISABLED // too stringent for this test
#include "gtest/gtest.h"

#include <common/defs.hpp>
#include <common/gt_assert.hpp>
#include <stencil-composition/backend.hpp>
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

namespace test_iterate_domain {

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > x_interval;
    typedef gridtools::interval< gridtools::level< 0, -2 >, gridtools::level< 1, 1 > > axis_t;

    typedef layout_map< 2, 1, 0 > layout_ijk_t;
    typedef layout_map< 0, 1, 2 > layout_kji_t;
    typedef layout_map< 0, 1 > layout_ij_t;

    typedef gridtools::backend< enumtype::Cuda, enumtype::structured, enumtype::Block > backend_t;
    typedef gridtools::cuda_storage_info< 0, layout_ijk_t > meta_ijk_t;
    typedef gridtools::cuda_storage_info< 0, layout_kji_t > meta_kji_t;
    typedef gridtools::cuda_storage_info< 0, layout_ij_t > meta_ij_t;

    typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< float_type, meta_ijk_t > storage_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< float_type, meta_kji_t > storage_buff_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< float_type, meta_ij_t > storage_out_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< bool, meta_ij_t > storage_bool_t;

    // These are the stencil operators that compose the multistage stencil in this test
    struct dummy_functor {
        typedef accessor< 0, enumtype::in, extent< 0, 0, 0, 0 > > read_only_texture_arg;
        typedef accessor< 1, enumtype::inout, extent< 0, 0, 0, 0 > > out;
        typedef accessor< 2, enumtype::in, extent< 0, 0, 0, 0 > > read_only_bypass_arg;
        typedef accessor< 3, enumtype::in, extent< 0, 0, 0, 0 > > read_only_non_texture_type_arg;
        typedef accessor< 4, enumtype::in, extent< -2, 1, -3, 4 > > shared_mem_arg;
        typedef accessor< 5, enumtype::inout, extent< 0, 0, 0, 0, -1, 2 > > kcache_arg;

        typedef boost::mpl::vector< read_only_texture_arg,
            out,
            read_only_bypass_arg,
            read_only_non_texture_type_arg,
            shared_mem_arg,
            kcache_arg > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {}
    };

    std::ostream &operator<<(std::ostream &s, dummy_functor const) { return s << "dummy_function"; }

} // namespace test_iterate_domain

TEST(test_iterate_domain, accessor_metafunctions) {

    using namespace test_iterate_domain;

    uint_t d1 = 15;
    uint_t d2 = 13;
    uint_t d3 = 18;

    meta_ijk_t meta_ijk_(d1, d2, d3);
    storage_t read_only_texture_arg(meta_ijk_, 0.0);
    meta_kji_t meta_kji_(d1, d2, d3);
    storage_buff_t read_only_bypass_arg(meta_kji_, 0.0);

    meta_ij_t meta_ij_(d1, d2);
    storage_out_t out(meta_ij_, 0.0);

    storage_bool_t read_only_non_texture_type_arg(meta_ij_, false);
    storage_t shared_mem_arg(meta_ijk_, 0.0);
    storage_t kcache_arg(meta_ijk_, 0.0);

    typedef arg< 0, storage_t > p_read_only_texture_arg;
    typedef arg< 1, storage_out_t > p_out;
    typedef arg< 2, storage_buff_t > p_read_only_bypass_arg;
    typedef arg< 3, storage_bool_t > p_read_only_non_texture_type_arg;
    typedef arg< 4, storage_t > p_shared_mem_arg;
    typedef arg< 5, storage_t > p_kcache_arg;

    halo_descriptor di{4, 4, 4, d1 - 4 - 1, d1};
    halo_descriptor dj{4, 4, 4, d2 - 4 - 1, d2};

    gridtools::grid< axis_t > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    using caches_t = decltype(define_caches(cache< bypass, cache_io_policy::local >(p_read_only_bypass_arg()),
        cache< IJ, cache_io_policy::local >(p_shared_mem_arg()),
        cache< K, cache_io_policy::local >(p_kcache_arg())));

    auto computation_ =
        gridtools::make_computation< backend_t >(grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                                                     caches_t(),
                                                     gridtools::make_stage< dummy_functor >(p_read_only_texture_arg(),
                                                         p_out(),
                                                         p_read_only_bypass_arg(),
                                                         p_read_only_non_texture_type_arg(),
                                                         p_shared_mem_arg(),
                                                         p_kcache_arg())));

    typedef decltype(gridtools::make_stage< dummy_functor >(p_read_only_texture_arg(),
        p_out(),
        p_read_only_bypass_arg(),
        p_read_only_non_texture_type_arg(),
        p_shared_mem_arg(),
        p_kcache_arg())) esf_t;

    typedef decltype(computation_) intermediate_t;
    typedef intermediate_mss_local_domains< intermediate_t > mss_local_domains_t;

    typedef boost::mpl::front< mss_local_domains_t >::type mss_local_domain1_t;

    typedef iterate_domain_cuda<
        iterate_domain,
        iterate_domain_arguments< backend_ids< Cuda, GRIDBACKEND, Block >,
            boost::mpl::at_c< typename mss_local_domain1_t::fused_local_domain_sequence_t, 0 >::type,
            boost::mpl::vector1< esf_t >,
            boost::mpl::vector1< extent< 0, 0, 0, 0 > >,
            extent< 1, -1, 1, -1 >,
            caches_t,
            block_size< 32, 4, 1 >,
            block_size< 32, 4, 1 >,
            gridtools::grid< axis_t >,
            boost::mpl::false_,
            notype > > it_domain_t;

    GRIDTOOLS_STATIC_ASSERT(
        (it_domain_t::template accessor_points_to_readonly_arg< dummy_functor::read_only_texture_arg >::type::value),
        "Error");
    GRIDTOOLS_STATIC_ASSERT(
        (it_domain_t::template accessor_points_to_readonly_arg< dummy_functor::read_only_bypass_arg >::type::value),
        "Error");

    GRIDTOOLS_STATIC_ASSERT(
        !(it_domain_t::template accessor_points_to_readonly_arg< dummy_functor::out >::type::value), "Error");

    GRIDTOOLS_STATIC_ASSERT(
        (it_domain_t::template accessor_read_from_texture< dummy_functor::read_only_texture_arg >::type::value),
        "Error");

    // because is output field
    GRIDTOOLS_STATIC_ASSERT(
        !(it_domain_t::template accessor_read_from_texture< dummy_functor::out >::type::value), "Error");
    // because is being bypass
    GRIDTOOLS_STATIC_ASSERT(
        !(it_domain_t::template accessor_read_from_texture< dummy_functor::read_only_bypass_arg >::type::value),
        "Error");
    // because is not a texture supported type
    GRIDTOOLS_STATIC_ASSERT(!(it_domain_t::template accessor_read_from_texture<
                                dummy_functor::read_only_non_texture_type_arg >::type::value),
        "Error");

    // access via shared mem
    GRIDTOOLS_STATIC_ASSERT(
        (it_domain_t::template accessor_from_shared_mem< dummy_functor::shared_mem_arg >::type::value), "Error");
    GRIDTOOLS_STATIC_ASSERT(
        !(it_domain_t::template accessor_from_shared_mem< dummy_functor::read_only_bypass_arg >::type::value), "Error");

    // access via kcache reg
    GRIDTOOLS_STATIC_ASSERT(
        (it_domain_t::template accessor_from_kcache_reg< dummy_functor::kcache_arg >::type::value), "Error");
    GRIDTOOLS_STATIC_ASSERT(
        !(it_domain_t::template accessor_from_kcache_reg< dummy_functor::shared_mem_arg >::type::value), "Error");
}
