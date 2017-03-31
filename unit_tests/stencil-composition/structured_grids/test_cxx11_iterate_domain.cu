/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "common/defs.hpp"
#include "stencil-composition/stencil-composition.hpp"

using namespace gridtools;
using namespace enumtype;

namespace test_iterate_domain {

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > x_interval;
    typedef gridtools::interval< gridtools::level< 0, -2 >, gridtools::level< 1, 1 > > axis;

    typedef layout_map< 2, 1, 0 > layout_ijk_t;
    typedef layout_map< 0, 1, 2 > layout_kji_t;
    typedef layout_map< 0, 1 > layout_ij_t;

    typedef gridtools::backend< enumtype::Cuda, enumtype::structured, enumtype::Block > backend_t;
    typedef backend_t::storage_info< 0, layout_ijk_t > meta_ijk_t;
    typedef backend_t::storage_info< 0, layout_kji_t > meta_kji_t;
    typedef backend_t::storage_info< 0, layout_ij_t > meta_ij_t;

    typedef backend_t::storage_type< float_type, meta_ijk_t >::type storage_type;
    typedef backend_t::storage_type< float_type, meta_kji_t >::type storage_buff_type;
    typedef backend_t::storage_type< float_type, meta_ij_t >::type storage_out_type;
    typedef backend_t::storage_type< bool, meta_ij_t >::type storage_bool_type;

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
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
    };

    std::ostream &operator<<(std::ostream &s, dummy_functor const) { return s << "dummy_function"; }

} // namespace test_iterate_domain

TEST(test_iterate_domain, accessor_metafunctions) {

    using namespace test_iterate_domain;

    uint_t d1 = 15;
    uint_t d2 = 13;
    uint_t d3 = 18;

    meta_ijk_t meta_ijk_(d1, d2, d3);
    storage_type read_only_texture_arg(meta_ijk_);
    meta_kji_t meta_kji_(d1, d2, d3);
    storage_buff_type read_only_bypass_arg(meta_kji_);

    meta_ij_t meta_ij_(d1, d2);
    storage_out_type out(meta_ij_);

    storage_bool_type read_only_non_texture_type_arg(meta_ij_);
    storage_type shared_mem_arg(meta_ijk_);
    storage_type kcache_arg(meta_ijk_);

    typedef arg< 0, storage_type > p_read_only_texture_arg;
    typedef arg< 1, storage_out_type > p_out;
    typedef arg< 2, storage_buff_type > p_read_only_bypass_arg;
    typedef arg< 3, storage_bool_type > p_read_only_non_texture_type_arg;
    typedef arg< 4, storage_type > p_shared_mem_arg;
    typedef arg< 5, storage_type > p_kcache_arg;

    typedef boost::mpl::vector< p_read_only_texture_arg,
        p_out,
        p_read_only_bypass_arg,
        p_read_only_non_texture_type_arg,
        p_shared_mem_arg,
        p_kcache_arg > accessor_list;

    gridtools::aggregator_type< accessor_list > domain((p_read_only_texture_arg() = read_only_texture_arg),
        (p_out() = out),
        (p_read_only_bypass_arg() = read_only_bypass_arg),
        (p_read_only_non_texture_type_arg() = read_only_non_texture_type_arg),
        (p_shared_mem_arg() = shared_mem_arg),
        (p_kcache_arg() = kcache_arg));

    uint_t di[5] = {4, 4, 4, d1 - 4 - 1, d1};
    uint_t dj[5] = {4, 4, 4, d2 - 4 - 1, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    using caches_t = decltype(define_caches(cache< bypass, local >(p_read_only_bypass_arg()),
        cache< IJ, local >(p_shared_mem_arg()),
        cache< K, local >(p_kcache_arg())));

    auto computation_ =
        gridtools::make_computation< backend_t >(domain,
            grid,
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

    typedef boost::remove_reference< decltype(*computation_) >::type intermediate_t;
    typedef intermediate_mss_local_domains< intermediate_t >::type mss_local_domains_t;

    typedef boost::mpl::front< mss_local_domains_t >::type mss_local_domain1_t;

    typedef iterate_domain_cuda<
        iterate_domain,
        iterate_domain_arguments< backend_ids< Cuda, GRIDBACKEND, Block >,
            boost::mpl::at_c< typename mss_local_domain1_t::fused_local_domain_sequence_t, 0 >::type,
            boost::mpl::vector1< esf_t >,
            boost::mpl::vector1< extent< 0, 0, 0, 0 > >,
            extent< 1, -1, 1, -1 >,
            caches_t,
            block_size< 32, 4, 0 >,
            block_size< 32, 4, 0 >,
            gridtools::grid< axis >,
            boost::mpl::false_,
            notype > > it_domain_t;

    static_assert(
        (it_domain_t::template accessor_points_to_readonly_arg< dummy_functor::read_only_texture_arg >::type::value),
        "Error");
    static_assert(
        (it_domain_t::template accessor_points_to_readonly_arg< dummy_functor::read_only_bypass_arg >::type::value),
        "Error");

    static_assert(!(it_domain_t::template accessor_points_to_readonly_arg< dummy_functor::out >::type::value), "Error");

    static_assert(
        (it_domain_t::template accessor_read_from_texture< dummy_functor::read_only_texture_arg >::type::value),
        "Error");

    // because is output field
    static_assert(!(it_domain_t::template accessor_read_from_texture< dummy_functor::out >::type::value), "Error");
    // because is being bypass
    static_assert(
        !(it_domain_t::template accessor_read_from_texture< dummy_functor::read_only_bypass_arg >::type::value),
        "Error");
    // because is not a texture supported type
    static_assert(!(it_domain_t::template accessor_read_from_texture<
                      dummy_functor::read_only_non_texture_type_arg >::type::value),
        "Error");

    // access via shared mem
    static_assert(
        (it_domain_t::template accessor_from_shared_mem< dummy_functor::shared_mem_arg >::type::value), "Error");
    static_assert(
        !(it_domain_t::template accessor_from_shared_mem< dummy_functor::read_only_bypass_arg >::type::value), "Error");

    // access via kcache reg
    static_assert((it_domain_t::template accessor_from_kcache_reg< dummy_functor::kcache_arg >::type::value), "Error");
    static_assert(
        !(it_domain_t::template accessor_from_kcache_reg< dummy_functor::shared_mem_arg >::type::value), "Error");

    ASSERT_TRUE(true);
}
