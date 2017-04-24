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
#include "gtest/gtest.h"
#include <gridtools.hpp>
#include "storage-facility.hpp"
#include "stencil-composition/stencil-composition.hpp"

/*
  This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO
 */

using namespace gridtools;
using namespace enumtype;

TEST(assign_placeholders, test) {

#ifdef CUDA_EXAMPLE
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3, halo< 1, 1, 1 > > storage_info1_t;
    typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3, halo< 2, 2, 2 > > storage_info2_t;
    typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< float_type, storage_info1_t >
        data_store1_t;
    typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< float_type, storage_info2_t >
        data_store2_t;

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    storage_info1_t meta_1(d1, d2, d3);
    storage_info2_t meta_2(d1, d2, d3);

    data_store1_t in(meta_1);
    data_store2_t out(meta_2);
    data_store2_t coeff(meta_2);

    in.allocate();
    out.allocate();
    coeff.allocate();

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef arg< 0, data_store1_t, default_location_type, true > p_lap;
    typedef arg< 1, data_store1_t, default_location_type, true > p_flx;
    typedef arg< 2, data_store2_t, default_location_type, true > p_fly;
    typedef arg< 3, data_store2_t > p_coeff;
    typedef arg< 4, data_store1_t > p_in;
    typedef arg< 5, data_store2_t > p_out;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector< p_lap, p_flx, p_fly, p_coeff, p_in, p_out > accessor_list;

    // printf("coeff (3) pointer: %x\n", &coeff);
    // printf("in    (4) pointer: %x\n", &in);
    // printf("out   (5) pointer: %x\n", &out);

    gridtools::aggregator_type< accessor_list > domain(coeff, in, out);

    using dst1_tmp = gridtools::data_store< gridtools::host_storage< double >,
        gridtools::host_storage_info< 1u,
                                                gridtools::layout_map< 0, 1, 2 >,
                                                gridtools::halo< 1u, 1u, 1u >,
                                                gridtools::alignment< 1u > > >;
    using dst1 = gridtools::data_store< gridtools::host_storage< double >,
        gridtools::host_storage_info< 0u,
                                            gridtools::layout_map< 0, 1, 2 >,
                                            gridtools::halo< 1u, 1u, 1u >,
                                            gridtools::alignment< 1u > > >;
    using dst2_tmp = gridtools::data_store< gridtools::host_storage< double >,
        gridtools::host_storage_info< 1u,
                                                gridtools::layout_map< 0, 1, 2 >,
                                                gridtools::halo< 2u, 2u, 2u >,
                                                gridtools::alignment< 1u > > >;
    using dst2 = gridtools::data_store< gridtools::host_storage< double >,
        gridtools::host_storage_info< 0u,
                                            gridtools::layout_map< 0, 1, 2 >,
                                            gridtools::halo< 2u, 2u, 2u >,
                                            gridtools::alignment< 1u > > >;

    // Check data store type correctness
    typedef typename boost::is_same<
        decltype(domain.m_arg_storage_pair_list),
        boost::fusion::vector6<
            gridtools::arg_storage_pair< gridtools::arg< 0u, dst1_tmp, default_location_type, true >, dst1_tmp >,
            gridtools::arg_storage_pair< gridtools::arg< 1u, dst1_tmp, default_location_type, true >, dst1_tmp >,
            gridtools::arg_storage_pair< gridtools::arg< 2u, dst2_tmp, default_location_type, true >, dst2_tmp >,
            gridtools::arg_storage_pair< gridtools::arg< 3u, dst2 >, dst2 >,
            gridtools::arg_storage_pair< gridtools::arg< 4u, dst1 >, dst1 >,
            gridtools::arg_storage_pair< gridtools::arg< 5u, dst2 >, dst2 > > >::type check_storages_t;
    static_assert(check_storages_t::value, "Type check failed.");

    // Check metadata_set correctness
    typedef typename boost::is_same< gridtools::metadata_set< boost::mpl::v_item<
                                         // temporary with halo size 1
                                         gridtools::pointer< const gridtools::host_storage_info< 1u,
                                             gridtools::layout_map< 0, 1, 2 >,
                                             gridtools::halo< 1u, 1u, 1u >,
                                             gridtools::alignment< 1u > > >,
                                         boost::mpl::v_item<
                                             // temporary with halo size 2
                                             gridtools::pointer< const gridtools::host_storage_info< 1u,
                                                 gridtools::layout_map< 0, 1, 2 >,
                                                 gridtools::halo< 2u, 2u, 2u >,
                                                 gridtools::alignment< 1u > > >,
                                             boost::mpl::v_item<
                                                 // non-temporary with halo size 1
                                                 gridtools::pointer< const gridtools::host_storage_info< 0u,
                                                     gridtools::layout_map< 0, 1, 2 >,
                                                     gridtools::halo< 2u, 2u, 2u >,
                                                     gridtools::alignment< 1u > > >,
                                                 boost::mpl::v_item<
                                                     // non-temporary with halo size 2
                                                     gridtools::pointer< const gridtools::host_storage_info< 0u,
                                                         gridtools::layout_map< 0, 1, 2 >,
                                                         gridtools::halo< 1u, 1u, 1u >,
                                                         gridtools::alignment< 1u > > >,
                                                     boost::mpl::vector0< mpl_::na >,
                                                     0 >,
                                                 0 >,
                                             0 >,
                                         0 > >,
        typename decltype(domain)::metadata_set_t >::type check_storage_infos_t;

    static_assert(check_storage_infos_t::value, "Type check failed.");

    // Check pointers
    assert(domain.template get_arg_storage_pair< p_flx >().ptr.get() == 0x0);
    assert(domain.template get_arg_storage_pair< p_fly >().ptr.get() == 0x0);
    assert(domain.template get_arg_storage_pair< p_lap >().ptr.get() == 0x0);
    assert(domain.template get_arg_storage_pair< p_coeff >().ptr.get() == &coeff);
    assert(domain.template get_arg_storage_pair< p_in >().ptr.get() == &in);
    assert(domain.template get_arg_storage_pair< p_out >().ptr.get() == &out);

    // Temporary storage info ptrs are not present yet
    assert(!(domain.get_metadata_set()
                 .template present< gridtools::pointer< const gridtools::host_storage_info< 1u,
                     gridtools::layout_map< 0, 1, 2 >,
                     gridtools::halo< 1u, 1u, 1u >,
                     gridtools::alignment< 1u > > > >()));
    assert(!(domain.get_metadata_set()
                 .template present< gridtools::pointer< const gridtools::host_storage_info< 1u,
                     gridtools::layout_map< 0, 1, 2 >,
                     gridtools::halo< 2u, 2u, 2u >,
                     gridtools::alignment< 1u > > > >()));

    // Non-temporary storage info ptrs are present
    assert((domain.get_metadata_set()
                .template present< gridtools::pointer< const gridtools::host_storage_info< 0u,
                    gridtools::layout_map< 0, 1, 2 >,
                    gridtools::halo< 1u, 1u, 1u >,
                    gridtools::alignment< 1u > > > >()));
    assert((domain.get_metadata_set()
                .template present< gridtools::pointer< const gridtools::host_storage_info< 0u,
                    gridtools::layout_map< 0, 1, 2 >,
                    gridtools::halo< 2u, 2u, 2u >,
                    gridtools::alignment< 1u > > > >()));
    assert((domain.get_metadata_set()
                .template get< gridtools::pointer< const gridtools::host_storage_info< 0u,
                    gridtools::layout_map< 0, 1, 2 >,
                    gridtools::halo< 1u, 1u, 1u >,
                    gridtools::alignment< 1u > > > >()
                .get() == in.get_storage_info_ptr()));
    assert((domain.get_metadata_set()
                .template get< gridtools::pointer< const gridtools::host_storage_info< 0u,
                    gridtools::layout_map< 0, 1, 2 >,
                    gridtools::halo< 2u, 2u, 2u >,
                    gridtools::alignment< 1u > > > >()
                .get() == out.get_storage_info_ptr()));

    // lets do a reassign
    storage_info1_t meta_1_new(d1, d2, d3);
    storage_info2_t meta_2_new(d1, d2, d3);

    data_store1_t in_new(meta_1_new);
    data_store2_t out_new(meta_2_new);
    data_store2_t coeff_new(meta_2_new);

    in_new.allocate();
    out_new.allocate();
    coeff_new.allocate();

    domain.reassign_impl(out_new, in_new, coeff_new);

    // check pointers again
    assert(domain.template get_arg_storage_pair< p_flx >().ptr.get() == 0x0);
    assert(domain.template get_arg_storage_pair< p_fly >().ptr.get() == 0x0);
    assert(domain.template get_arg_storage_pair< p_lap >().ptr.get() == 0x0);
    assert(domain.template get_arg_storage_pair< p_coeff >().ptr.get() == &out_new);
    assert(domain.template get_arg_storage_pair< p_in >().ptr.get() == &in_new);
    assert(domain.template get_arg_storage_pair< p_out >().ptr.get() == &coeff_new);

    // Temporary storage info ptrs are not present yet
    assert(!(domain.get_metadata_set()
                 .template present< gridtools::pointer< const gridtools::host_storage_info< 1u,
                     gridtools::layout_map< 0, 1, 2 >,
                     gridtools::halo< 1u, 1u, 1u >,
                     gridtools::alignment< 1u > > > >()));
    assert(!(domain.get_metadata_set()
                 .template present< gridtools::pointer< const gridtools::host_storage_info< 1u,
                     gridtools::layout_map< 0, 1, 2 >,
                     gridtools::halo< 2u, 2u, 2u >,
                     gridtools::alignment< 1u > > > >()));

    // Non-temporary storage info ptrs are present
    assert((domain.get_metadata_set()
                .template present< gridtools::pointer< const gridtools::host_storage_info< 0u,
                    gridtools::layout_map< 0, 1, 2 >,
                    gridtools::halo< 1u, 1u, 1u >,
                    gridtools::alignment< 1u > > > >()));
    assert((domain.get_metadata_set()
                .template present< gridtools::pointer< const gridtools::host_storage_info< 0u,
                    gridtools::layout_map< 0, 1, 2 >,
                    gridtools::halo< 2u, 2u, 2u >,
                    gridtools::alignment< 1u > > > >()));

    assert((domain.get_metadata_set()
                .template get< gridtools::pointer< const gridtools::host_storage_info< 0u,
                    gridtools::layout_map< 0, 1, 2 >,
                    gridtools::halo< 1u, 1u, 1u >,
                    gridtools::alignment< 1u > > > >()
                .get() == in_new.get_storage_info_ptr()));
    assert((domain.get_metadata_set()
                .template get< gridtools::pointer< const gridtools::host_storage_info< 0u,
                    gridtools::layout_map< 0, 1, 2 >,
                    gridtools::halo< 2u, 2u, 2u >,
                    gridtools::alignment< 1u > > > >()
                .get() == coeff_new.get_storage_info_ptr()));
}
