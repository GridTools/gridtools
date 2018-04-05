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

#include "common/defs.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "stencil-composition/structured_grids/accessor.hpp"
#include "gtest/gtest.h"
#include <iostream>

#include "backend_select.hpp"

namespace test_iterate_domain {
    using namespace gridtools;
    using namespace enumtype;

    // These are the stencil operators that compose the multistage stencil in this test
    struct dummy_functor {
        typedef accessor< 0, enumtype::in, extent< 0, 0, 0, 0 >, 6 > in;
        typedef accessor< 1, enumtype::in, extent< 0, 0, 0, 0 >, 5 > buff;
        typedef accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 > out;
        typedef boost::mpl::vector< in, buff, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {}
    };

    std::ostream &operator<<(std::ostream &s, dummy_functor const) { return s << "dummy_function"; }

    bool static test() {
        typedef layout_map< 3, 2, 1, 0 > layout_ijkp_t;
        typedef layout_map< 0, 1, 2 > layout_kji_t;
        typedef layout_map< 0, 1 > layout_ij_t;

        typedef backend_traits_from_id< backend_t::s_backend_id > backend_traits_t;
        typedef storage_traits< backend_t::s_backend_id > storage_traits_t;
        typedef typename storage_traits_t::custom_layout_storage_info_t< 0, layout_ijkp_t > meta_ijkp_t;
        typedef typename storage_traits_t::custom_layout_storage_info_t< 0, layout_kji_t > meta_kji_t;
        typedef typename storage_traits_t::custom_layout_storage_info_t< 0, layout_ij_t > meta_ij_t;

        typedef gridtools::storage_traits<
            backend_t::s_backend_id >::data_store_field_t< float_type, meta_ijkp_t, 3, 2, 1 > storage_t;
        typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_field_t< float_type, meta_kji_t, 4, 7 >
            storage_buff_t;
        typedef gridtools::storage_traits<
            backend_t::s_backend_id >::data_store_field_t< float_type, meta_ij_t, 2, 2, 2 > storage_out_t;

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

        typedef arg< 0, storage_t > p_in;
        typedef arg< 1, storage_buff_t > p_buff;
        typedef arg< 2, storage_out_t > p_out;
        typedef boost::mpl::vector< p_in, p_buff, p_out > accessor_list;

        gridtools::aggregator_type< accessor_list > domain(in, buff, out);

        auto grid = make_grid(d1, d2, d3);

        auto mss_ = gridtools::make_multistage // mss_descriptor
            (enumtype::execute< enumtype::forward >(),
                gridtools::make_stage< dummy_functor >(p_in(), p_buff(), p_out()));
        auto computation_ = make_computation< gridtools::backend< Host, GRIDBACKEND, Naive > >(domain, grid, mss_);

        typedef decltype(gridtools::make_stage< dummy_functor >(p_in(), p_buff(), p_out())) esf_t;

        computation_->ready();
        computation_->steady();

        typedef boost::remove_reference< decltype(*computation_) >::type intermediate_t;
        typedef intermediate_mss_local_domains< intermediate_t > mss_local_domains_t;

        typedef boost::mpl::front< mss_local_domains_t >::type mss_local_domain1_t;

        typedef typename backend_traits_t::select_iterate_domain<
            iterate_domain_arguments< backend_ids< Host, GRIDBACKEND, Naive >,
                boost::mpl::at_c< typename mss_local_domain1_t::fused_local_domain_sequence_t, 0 >::type,
                boost::mpl::vector1< esf_t >,
                boost::mpl::vector1< extent< 0, 0, 0, 0 > >,
                extent< 0, 0, 0, 0 >,
                boost::mpl::vector0<>,
                block_size< 32, 4 >,
                block_size< 32, 4 >,
                gridtools::grid< gridtools::axis< 1 >::axis_interval_t >,
                boost::mpl::false_,
                notype > >::type it_domain_t;

        mss_local_domain1_t mss_local_domain1 = boost::fusion::at_c< 0 >(computation_->mss_local_domain_list());
        auto local_domain1 = boost::fusion::at_c< 0 >(mss_local_domain1.local_domain_list);
        it_domain_t it_domain(local_domain1, 0);

        GRIDTOOLS_STATIC_ASSERT(it_domain_t::N_STORAGES == 3, "bug in iterate domain, incorrect number of storages");

        GRIDTOOLS_STATIC_ASSERT(
            it_domain_t::N_DATA_POINTERS == 23, "bug in iterate domain, incorrect number of data pointers");

#ifdef BACKEND_MIC
        auto const &data_pointer = it_domain.data_pointer();
#else
        typename it_domain_t::data_ptr_cached_t data_pointer;
        typedef typename it_domain_t::strides_cached_t strides_t;
        strides_t strides;

        it_domain.set_data_pointer_impl(&data_pointer);
        it_domain.set_strides_pointer_impl(&strides);

        it_domain.template assign_storage_pointers< backend_traits_t >();
        it_domain.template assign_stride_pointers< backend_traits_t, strides_t >();
#endif

        // check data pointers initialization
        assert(
            ((float_type *)data_pointer.template get< 0 >()[0] == in.get< 0, 0 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 0 >()[1] == in.get< 0, 1 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 0 >()[2] == in.get< 0, 2 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 0 >()[3] == in.get< 1, 0 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 0 >()[4] == in.get< 1, 1 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 0 >()[5] == in.get< 2, 0 >().get_storage_ptr()->get_cpu_ptr()));

        assert(
            ((float_type *)data_pointer.template get< 1 >()[0] == buff.get< 0, 0 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 1 >()[1] == buff.get< 0, 1 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 1 >()[2] == buff.get< 0, 2 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 1 >()[3] == buff.get< 0, 3 >().get_storage_ptr()->get_cpu_ptr()));

        assert(
            ((float_type *)data_pointer.template get< 1 >()[4] == buff.get< 1, 0 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 1 >()[5] == buff.get< 1, 1 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 1 >()[6] == buff.get< 1, 2 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 1 >()[7] == buff.get< 1, 3 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 1 >()[8] == buff.get< 1, 4 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 1 >()[9] == buff.get< 1, 5 >().get_storage_ptr()->get_cpu_ptr()));
        assert((
            (float_type *)data_pointer.template get< 1 >()[10] == buff.get< 1, 6 >().get_storage_ptr()->get_cpu_ptr()));

        assert(
            ((float_type *)data_pointer.template get< 2 >()[0] == out.get< 0, 0 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 2 >()[1] == out.get< 0, 1 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 2 >()[2] == out.get< 1, 0 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 2 >()[3] == out.get< 1, 1 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 2 >()[4] == out.get< 2, 0 >().get_storage_ptr()->get_cpu_ptr()));
        assert(
            ((float_type *)data_pointer.template get< 2 >()[5] == out.get< 2, 1 >().get_storage_ptr()->get_cpu_ptr()));
// check field storage access

// using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
#ifndef BACKEND_MIC
        it_domain.reset_index();
#endif
        auto inv = make_field_host_view(in);
        inv.get< 0, 0 >()(0, 0, 0, 0) = 0.; // is accessor<0>
        inv.get< 0, 1 >()(0, 0, 0, 0) = 1.;
        inv.get< 0, 2 >()(0, 0, 0, 0) = 2.;
        inv.get< 1, 0 >()(0, 0, 0, 0) = 10.;
        inv.get< 1, 1 >()(0, 0, 0, 0) = 11.;
        inv.get< 2, 0 >()(0, 0, 0, 0) = 20.;

        assert(
            it_domain(alias< inout_accessor< 0, extent< 0, 0, 0, 0, 0, 0 >, 6 >, dimension< 6 > >::set< 0 >()) == 0.);
        assert(
            it_domain(alias< inout_accessor< 0, extent< 0, 0, 0, 0, 0, 0 >, 6 >, dimension< 6 > >::set< 1 >()) == 1.);
        assert(
            it_domain(alias< inout_accessor< 0, extent< 0, 0, 0, 0, 0, 0 >, 6 >, dimension< 6 > >::set< 2 >()) == 2.);
        assert(
            it_domain(alias< inout_accessor< 0, extent< 0, 0, 0, 0, 0, 0 >, 6 >, dimension< 5 > >::set< 1 >()) == 10.);
        assert(it_domain(
                   alias< inout_accessor< 0, extent< 0, 0, 0, 0, 0, 0 >, 6 >, dimension< 6 >, dimension< 5 > >::set< 1,
                       1 >()) == 11.);
        assert(
            it_domain(alias< inout_accessor< 0, extent< 0, 0, 0, 0, 0, 0 >, 6 >, dimension< 5 > >::set< 2 >()) == 20.);

        // using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
        auto buffv = make_field_host_view(buff);
        buffv.get< 0, 0 >()(0, 0, 0) = 0.; // is accessor<1>
        buffv.get< 0, 1 >()(0, 0, 0) = 1.;
        buffv.get< 0, 2 >()(0, 0, 0) = 2.;
        buffv.get< 0, 3 >()(0, 0, 0) = 3.;
        buffv.get< 1, 0 >()(0, 0, 0) = 10.;
        buffv.get< 1, 1 >()(0, 0, 0) = 11.;
        buffv.get< 1, 2 >()(0, 0, 0) = 12.;
        buffv.get< 1, 3 >()(0, 0, 0) = 13.;
        buffv.get< 1, 4 >()(0, 0, 0) = 14.;
        buffv.get< 1, 5 >()(0, 0, 0) = 15.;
        buffv.get< 1, 6 >()(0, 0, 0) = 16.;

        assert(it_domain(
                   alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >, dimension< 5 > >::set< 0 >()) == 0.);
        assert(it_domain(
                   alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >, dimension< 5 > >::set< 1 >()) == 1.);
        assert(it_domain(
                   alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >, dimension< 5 > >::set< 2 >()) == 2.);
        assert(
            it_domain(alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >, dimension< 4 > >::set< 1 >()) ==
            10.);
        assert(it_domain(alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >,
                   dimension< 4 >,
                   dimension< 5 > >::set< 1, 1 >()) == 11.);
        assert(it_domain(alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >,
                   dimension< 4 >,
                   dimension< 5 > >::set< 1, 2 >()) == 12.);
        assert(it_domain(alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >,
                   dimension< 4 >,
                   dimension< 5 > >::set< 1, 3 >()) == 13.);
        assert(it_domain(alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >,
                   dimension< 4 >,
                   dimension< 5 > >::set< 1, 4 >()) == 14.);
        assert(it_domain(alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >,
                   dimension< 4 >,
                   dimension< 5 > >::set< 1, 5 >()) == 15.);
        assert(it_domain(alias< accessor< 1, enumtype::in, extent< 0, 0, 0, 0, 0 >, 5 >,
                   dimension< 4 >,
                   dimension< 5 > >::set< 1, 6 >()) == 16.);

        auto outv = make_field_host_view(out);
        outv.get< 0, 0 >()(0, 0) = 0.; // is accessor<2>
        outv.get< 0, 1 >()(0, 0) = 1.;
        outv.get< 1, 0 >()(0, 0) = 10.;
        outv.get< 1, 1 >()(0, 0) = 11.;
        outv.get< 2, 0 >()(0, 0) = 20.;
        outv.get< 2, 1 >()(0, 0) = 21.;

        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >()) == 0.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(dimension< 4 >(1))) == 1.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(dimension< 3 >(1))) == 10.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(
                   dimension< 4 >(1), dimension< 3 >(1))) == 11.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(dimension< 3 >(2))) == 20.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(
                   dimension< 3 >(2), dimension< 4 >(1))) == 21.);

        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 0, 0)) == 0.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 0, 1)) == 1.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 1, 0)) == 10.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 1, 1)) == 11.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 2, 0)) == 20.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 2, 1)) == 21.);

        // check index initialization and increment

        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >()) == 0.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(dimension< 4 >(1))) == 1.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(dimension< 3 >(1))) == 10.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(
                   dimension< 4 >(1), dimension< 3 >(1))) == 11.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(dimension< 3 >(2))) == 20.);
        assert(it_domain(accessor< 2, enumtype::inout, extent< 0, 0, 0, 0 >, 4 >(
                   dimension< 3 >(2), dimension< 4 >(1))) == 21.);

        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 0, 0)) == 0.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 0, 1)) == 1.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 1, 0)) == 10.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 1, 1)) == 11.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 2, 0)) == 20.);
        assert(it_domain(inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >(0, 0, 2, 1)) == 21.);

        // check index initialization and increment

        array< int_t, 3 > index;
        index = it_domain.index();
        assert(index[0] == 0 && index[1] == 0 && index[2] == 0);
#ifndef BACKEND_MIC
        index[0] += 3;
        index[1] += 2;
        index[2] += 1;
        it_domain.set_index(index);

        index = it_domain.index();
        assert(index[0] == 3 && index[1] == 2 && index[2] == 1);
#endif

        auto mdo = out.template get< 0, 0 >().get_storage_info_ptr();
        auto mdb = buff.template get< 0, 0 >().get_storage_info_ptr();
        auto mdi = in.template get< 0, 0 >().get_storage_info_ptr();

        array< int_t, 3 > new_index;
#ifdef BACKEND_MIC
        it_domain.set_i_block_index(1);
        it_domain.set_j_block_index(1);
        it_domain.set_k_block_index(1);
#else
        it_domain.increment< 0, static_uint< 1 > >(); // increment i
        it_domain.increment< 1, static_uint< 1 > >(); // increment j
        it_domain.increment< 2, static_uint< 1 > >(); // increment k
#endif
        new_index = it_domain.index();

        // even thought the first case is 4D, we incremented only i,j,k, thus in the check below we don't need the extra
        // stride
        assert(index[0] + mdi->template stride< 0 >() + mdi->template stride< 1 >() + mdi->template stride< 2 >() ==
               new_index[0]);

        assert(index[1] + mdb->template stride< 0 >() + mdb->template stride< 1 >() + mdb->template stride< 2 >() ==
               new_index[1]);

        assert(index[2] + mdo->template stride< 0 >() + mdo->template stride< 1 >() == new_index[2]);

        // check offsets for the space dimensions
        using in_1_1 = alias< accessor< 0, enumtype::inout, extent< 0, 0, 0, 0, 0, 0 >, 6 >,
            dimension< 6 >,
            dimension< 5 > >::set< 1, 1 >;

        auto d1_ = in_1_1{dimension< 1 >{1}};
        auto d2_ = in_1_1{dimension< 2 >{1}};
        auto d3_ = in_1_1{dimension< 3 >{1}};
        auto d4_ = in_1_1{dimension< 4 >{1}};
        assert(((float_type *)(&inv.get< 1, 1 >()(0, 0, 0, 0) + new_index[0] + mdi->template stride< 0 >() ==
                               &it_domain(d1_))));

        assert(((float_type *)(&inv.get< 1, 1 >()(0, 0, 0, 0) + new_index[0] + mdi->template stride< 1 >() ==
                               &it_domain(d2_))));

        assert(((float_type *)(&inv.get< 1, 1 >()(0, 0, 0, 0) + new_index[0] + mdi->template stride< 2 >() ==
                               &it_domain(d3_))));

        assert(((float_type *)(&inv.get< 1, 1 >()(0, 0, 0, 0) + new_index[0] + mdi->template stride< 3 >() ==
                               &it_domain(d4_))));

        // check offsets for the space dimensions
        using buff_1_1 =
            alias< accessor< 1, enumtype::inout, extent< 0, 0, 0, 0, 0 >, 5 >, dimension< 5 >, dimension< 4 > >::set< 1,
                1 >;
        auto b1_ = buff_1_1{dimension< 1 >{1}};
        auto b2_ = buff_1_1{dimension< 2 >{1}};
        auto b3_ = buff_1_1{dimension< 3 >{1}};

        assert(((float_type *)(&buffv.get< 1, 1 >()(0, 0, 0) + new_index[1] + mdb->template stride< 0 >() ==
                               &it_domain(b1_))));

        assert(((float_type *)(&buffv.get< 1, 1 >()(0, 0, 0) + new_index[1] + mdb->template stride< 1 >() ==
                               &it_domain(b2_))));

        assert(((float_type *)(&buffv.get< 1, 1 >()(0, 0, 0) + new_index[1] + mdb->template stride< 2 >() ==
                               &it_domain(b3_))));

        using out_1 =
            alias< inout_accessor< 2, extent< 0, 0, 0, 0 >, 4 >, dimension< 4 >, dimension< 3 > >::set< 1, 1 >;

        auto c1_ = out_1{dimension< 1 >{1}};
        auto c2_ = out_1{dimension< 2 >{1}};

        assert((
            (float_type *)(&outv.get< 1, 1 >()(0, 0) + new_index[2] + mdo->template stride< 0 >() == &it_domain(c1_))));

        assert((
            (float_type *)(&outv.get< 1, 1 >()(0, 0) + new_index[2] + mdo->template stride< 1 >() == &it_domain(c2_))));

#ifndef BACKEND_MIC
        // check strides initialization
        // the layout is <3,2,1,0>, so we don't care about the stride<0> (==1) but the rest is checked.
        assert(mdi->template stride< 3 >() == strides.get< 0 >()[0]);
        assert(mdi->template stride< 2 >() == strides.get< 0 >()[1]);
        assert(mdi->template stride< 1 >() == strides.get< 0 >()[2]); // 4D storage

        assert(mdb->template stride< 0 >() == strides.get< 1 >()[0]);
        assert(mdb->template stride< 1 >() == strides.get< 1 >()[1]); // 3D storage

        assert(mdo->template stride< 0 >() == strides.get< 2 >()[0]); // 2D storage
#endif

        return true;
    }
} // namespace test_iterate_domain

TEST(testdomain, iterate_domain) { EXPECT_EQ(test_iterate_domain::test(), true); }
