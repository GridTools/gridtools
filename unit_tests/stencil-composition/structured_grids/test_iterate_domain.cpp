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

#ifdef BACKEND_X86
#include <gridtools/stencil-composition/structured_grids/backend_x86/iterate_domain_x86.hpp>
#endif

#ifdef BACKEND_MC
#include <gridtools/stencil-composition/structured_grids/backend_mc/iterate_domain_mc.hpp>
#endif

#include <tuple>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/structured_grids/accessor.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace gridtools {
    namespace {

        using in_acc = in_accessor<0, extent<>, 4>;
        using buff_acc = in_accessor<1>;
        using out_acc = inout_accessor<2, extent<>, 2>;

        struct dummy_functor {
            using arg_list = make_arg_list<in_acc, buff_acc, out_acc>;
            template <typename Evaluation>
            GT_FUNCTION static void Do(Evaluation &eval);
        };

        using layout_ijkp_t = layout_map<3, 2, 1, 0>;
        using layout_kji_t = layout_map<0, 1, 2>;
        using layout_ij_t = layout_map<0, 1>;

        using backend_traits_t = backend_traits_from_id<backend_t::backend_id_t>;
        using storage_traits_t = storage_traits<backend_t::backend_id_t>;

        using meta_ijkp_t = storage_traits_t::custom_layout_storage_info_t<0, layout_ijkp_t>;
        using meta_kji_t = storage_traits_t::custom_layout_storage_info_t<0, layout_kji_t>;
        using meta_ij_t = storage_traits_t::custom_layout_storage_info_t<0, layout_ij_t>;

        using storage_t = gridtools::storage_traits<backend_t::backend_id_t>::data_store_t<float_type, meta_ijkp_t>;
        using storage_buff_t = gridtools::storage_traits<backend_t::backend_id_t>::data_store_t<float_type, meta_kji_t>;
        using storage_out_t = gridtools::storage_traits<backend_t::backend_id_t>::data_store_t<float_type, meta_ij_t>;

        TEST(testdomain, iterate_domain) {

            uint_t d1 = 15;
            uint_t d2 = 13;
            uint_t d3 = 18;
            uint_t d4 = 6;

            meta_ijkp_t meta_ijkp_(d1 + 3, d2 + 2, d3 + 1, d4);
            storage_t in(meta_ijkp_);
            meta_kji_t meta_kji_(d1, d2, d3);
            storage_buff_t buff(meta_kji_);
            meta_ij_t meta_ij_(d1 + 2, d2 + 1);
            storage_out_t out(meta_ij_);

            arg<0, storage_t> p_in;
            arg<1, storage_buff_t> p_buff;
            arg<2, storage_out_t> p_out;

            auto grid = make_grid(d1, d2, d3);

            auto mss_ = gridtools::make_multistage // mss_descriptor
                (execute<execution::forward>(), gridtools::make_stage<dummy_functor>(p_in, p_buff, p_out));
            auto computation_ = make_computation<gridtools::backend<target::x86, grid_type_t, strategy::naive>>(
                grid, p_in = in, p_buff = buff, p_out = out, mss_);
            auto local_domain1 = std::get<0>(computation_.local_domains());

            using esf_t = decltype(gridtools::make_stage<dummy_functor>(p_in, p_buff, p_out));

            using iterate_domain_arguments_t =
                iterate_domain_arguments<backend_ids<target::x86, grid_type_t, strategy::naive>,
                    decltype(local_domain1),
                    make_arg_list<esf_t>,
                    make_arg_list<extent<>>,
                    extent<>,
                    make_arg_list<>,
                    gridtools::grid<gridtools::axis<1>::axis_interval_t>>;

#ifdef BACKEND_MC
            using it_domain_t = iterate_domain_mc<iterate_domain_arguments_t>;
#endif

#ifdef BACKEND_X86
            using it_domain_t = iterate_domain_x86<iterate_domain_arguments_t>;
#endif

            it_domain_t it_domain(local_domain1);

#ifndef BACKEND_MC
            typedef typename it_domain_t::strides_cached_t strides_t;
            strides_t strides;

            it_domain.set_strides_pointer_impl(&strides);

            it_domain.template assign_stride_pointers<backend_traits_t, strides_t>();
#endif

// using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
#ifndef BACKEND_MC
            it_domain.initialize({}, {}, {});
#endif
            auto inv = make_host_view(in);
            inv(0, 0, 0, 0) = 0.; // is accessor<0>

            EXPECT_EQ(0, (it_domain.deref<decltype(p_in), intent::in>(in_acc())));

            // using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
            auto buffv = make_host_view(buff);
            buffv(0, 0, 0) = 0.; // is accessor<1>

            EXPECT_EQ(0, (it_domain.deref<decltype(p_buff), intent::in>(buff_acc())));

            auto outv = make_host_view(out);
            outv(0, 0) = 0.; // is accessor<2>

            EXPECT_EQ(0, (it_domain.deref<decltype(p_out), intent::inout>(out_acc())));
            EXPECT_EQ(0, (it_domain.deref<decltype(p_out), intent::inout>(out_acc(0, 0))));

            // check index initialization and increment

            auto index = it_domain.index();
            ASSERT_EQ(0, index[0]);
            ASSERT_EQ(0, index[1]);
            ASSERT_EQ(0, index[2]);
#ifndef BACKEND_MC
            index[0] += 3;
            index[1] += 2;
            index[2] += 1;
            it_domain.set_index(index);

            index = it_domain.index();
            EXPECT_EQ(3, index[0]);
            EXPECT_EQ(2, index[1]);
            EXPECT_EQ(1, index[2]);
#endif

            auto mdo = out.get_storage_info_ptr();
            auto mdb = buff.get_storage_info_ptr();
            auto mdi = in.get_storage_info_ptr();

#ifdef BACKEND_MC
            it_domain.set_i_block_index(1);
            it_domain.set_j_block_index(1);
            it_domain.set_k_block_index(1);
#else
            it_domain.increment_i();
            it_domain.increment_j();
            it_domain.increment_k();
#endif
            auto new_index = it_domain.index();

            // even thought the first case is 4D, we incremented only i,j,k, thus in the check below we don't need the
            // extra stride
            EXPECT_EQ(index[0] + mdi->stride<0>() + mdi->stride<1>() + mdi->stride<2>(), new_index[0]);
            EXPECT_EQ(index[1] + mdb->stride<0>() + mdb->stride<1>() + mdb->stride<2>(), new_index[1]);
            EXPECT_EQ(index[2] + mdo->stride<0>() + mdo->stride<1>(), new_index[2]);

#ifndef BACKEND_MC
            // check strides initialization
            // the layout is <3,2,1,0>, so we don't care about the stride<0> (==1) but the rest is checked.
            EXPECT_EQ(mdi->stride<3>(), strides.get<0>()[0]);
            EXPECT_EQ(mdi->stride<2>(), strides.get<0>()[1]);
            EXPECT_EQ(mdi->stride<1>(), strides.get<0>()[2]); // 4D storage

            EXPECT_EQ(mdb->stride<0>(), strides.get<1>()[0]);
            EXPECT_EQ(mdb->stride<1>(), strides.get<1>()[1]); // 3D storage

            EXPECT_EQ(mdo->stride<0>(), strides.get<2>()[0]); // 2D storage
#endif
        }
    } // namespace
} // namespace gridtools
