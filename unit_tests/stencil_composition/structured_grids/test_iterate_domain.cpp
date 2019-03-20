/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#define GT_PEDANTIC_DISABLED // too stringent for this test

#ifdef GT_BACKEND_X86
#include <gridtools/stencil_composition/structured_grids/backend_x86/iterate_domain_x86.hpp>
#endif

#ifdef GT_BACKEND_MC
#include <gridtools/stencil_composition/structured_grids/backend_mc/iterate_domain_mc.hpp>
#endif

#include <tuple>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/backend.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/structured_grids/accessor.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace gridtools {
    namespace {

        using in_acc = in_accessor<0, extent<>, 4>;
        using buff_acc = in_accessor<1>;
        using out_acc = inout_accessor<2, extent<>, 2>;

        struct dummy_functor {
            using param_list = make_param_list<in_acc, buff_acc, out_acc>;
            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval);
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
                (execute::forward(), gridtools::make_stage<dummy_functor>(p_in, p_buff, p_out));
            auto computation_ = make_computation<gridtools::backend<target::x86, strategy::naive>>(
                grid, p_in = in, p_buff = buff, p_out = out, mss_);
            auto local_domain1 = std::get<0>(computation_.local_domains());

            using esf_t = decltype(gridtools::make_stage<dummy_functor>(p_in, p_buff, p_out));

            using iterate_domain_arguments_t = iterate_domain_arguments<backend_ids<target::x86, strategy::naive>,
                decltype(local_domain1),
                std::tuple<esf_t>,
                std::tuple<>,
                gridtools::grid<gridtools::axis<1>::axis_interval_t>>;

#ifdef GT_BACKEND_MC
            using it_domain_t = iterate_domain_mc<iterate_domain_arguments_t>;
#endif

#ifdef GT_BACKEND_X86
            using it_domain_t = iterate_domain_x86<iterate_domain_arguments_t>;
#endif

            it_domain_t it_domain(local_domain1);

#ifndef GT_BACKEND_MC

            it_domain.assign_stride_pointers();
#endif

// using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
#ifndef GT_BACKEND_MC
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
#ifndef GT_BACKEND_MC
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

#ifdef GT_BACKEND_MC
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
        }
    } // namespace
} // namespace gridtools
