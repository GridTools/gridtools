/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "gtest/gtest.h"
#include <gridtools.hpp>
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

    //    typedef gridtools::STORAGE<double, gridtools::layout_map<0,1,2> > storage_type;

    typedef gridtools::BACKEND::storage_type<float_type, gridtools::BACKEND::storage_info<0, gridtools::layout_map<0,1,2> > >::type storage_type;
    typedef gridtools::BACKEND::temporary_storage_type<float_type, gridtools::BACKEND::storage_info<0, gridtools::layout_map<0,1,2> > >::type tmp_storage_type;

    uint_t d1=5;
    uint_t d2=5;
    uint_t d3=5;


    storage_type::storage_info_type meta_(d1,d2,d3);

    storage_type in(meta_,-1., "in");
    storage_type out(meta_,-7.3, "out");
    storage_type coeff(meta_,8., "coeff");

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef arg<0, tmp_storage_type > p_lap;
    typedef arg<1, tmp_storage_type > p_flx;
    typedef arg<2, tmp_storage_type > p_fly;
    typedef arg<3, storage_type > p_coeff;
    typedef arg<4, storage_type > p_in;
    typedef arg<5, storage_type > p_out;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_lap, p_flx, p_fly, p_coeff, p_in, p_out> accessor_list;

    // printf("coeff (3) pointer: %x\n", &coeff);
    // printf("in    (4) pointer: %x\n", &in);
    // printf("out   (5) pointer: %x\n", &out);

    gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&coeff, &in, &out));

    ASSERT_TRUE((boost::fusion::at_c<3>(domain.storage_pointers_view()).get() == &coeff ) &&
                (boost::fusion::at_c<4>(domain.storage_pointers_view()).get() == &in) &&
                (boost::fusion::at_c<5>(domain.storage_pointers_view()).get() == &out));
}
