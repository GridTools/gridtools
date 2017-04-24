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

#include <iostream>

#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND_V Cuda
typedef backend< BACKEND_V, GRIDBACKEND, Block > be;
#else
#ifdef BACKEND_BLOCK
#define BACKEND_V Host
typedef backend< BACKEND_V, GRIDBACKEND, Block > be;
#else
#define BACKEND_V Host
typedef backend< BACKEND_V, GRIDBACKEND, Naive > be;
#endif
#endif

typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;
typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;

struct A {
    typedef accessor_extend< accessor< 0, in, extent<>, 3 >, 2 >::type pin;
    typedef accessor_extend< accessor< 1, inout, extent<>, 3 >, 2 >::type pout;
    typedef dimension< 4 > comp;
    typedef dimension< 5 > snap;
    typedef boost::mpl::vector< pin, pout > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        // copy first component elements
        eval(pout(comp(0), snap(0))) = eval(pin(comp(0), snap(0)));
        // copy second component elements
        eval(pout(comp(1), snap(0))) = eval(pin(comp(1), snap(0)));
        eval(pout(comp(1), snap(1))) = eval(pin(comp(1), snap(1)));
        // copy third component elements
        eval(pout(comp(2), snap(0))) = eval(pin(comp(2), snap(0)));
        eval(pout(comp(2), snap(1))) = eval(pin(comp(2), snap(1)));
        eval(pout(comp(2), snap(2))) = eval(pin(comp(2), snap(2)));
    }
};

int main() {
    int d1 = 128;
    int d2 = 128;
    int d3 = 80;

    typedef storage_traits< BACKEND_V >::storage_info_t< 0, 3 > storage_info_ty;                   // storage info type
    typedef storage_traits< BACKEND_V >::data_store_t< float_type, storage_info_ty > data_store_t; // data store type
    typedef storage_traits< BACKEND_V >::data_store_field_t< float_type, storage_info_ty, 1, 2, 3 >
        data_store_field_t; // data store field type with 3 components with size 1, 2, 3

    storage_info_ty si(d1, d2, d3);
    data_store_field_t dsf_in(si);
    data_store_field_t dsf_out(si);

    dsf_in.allocate();
    dsf_out.allocate();

    auto hv_in = make_field_host_view(dsf_in);
    auto hv_out = make_field_host_view(dsf_out);

    // fill with values
    unsigned x = 0;
    for (int i = 0; i < d1; ++i)
        for (int j = 0; j < d2; ++j)
            for (int k = 0; k < d3; ++k) {
                hv_in.template get< 0, 0 >()(i, j, k) = 1;
                hv_in.template get< 1, 0 >()(i, j, k) = 2;
                hv_in.template get< 1, 1 >()(i, j, k) = 3;
                hv_in.template get< 2, 0 >()(i, j, k) = 4;
                hv_in.template get< 2, 1 >()(i, j, k) = 5;
                hv_in.template get< 2, 2 >()(i, j, k) = 6;

                hv_out.template get< 0, 0 >()(i, j, k) = 123;
                hv_out.template get< 1, 0 >()(i, j, k) = 123;
                hv_out.template get< 1, 1 >()(i, j, k) = 123;
                hv_out.template get< 2, 0 >()(i, j, k) = 123;
                hv_out.template get< 2, 1 >()(i, j, k) = 123;
                hv_out.template get< 2, 2 >()(i, j, k) = 123;
            }

    // create some gridtools stuff
    typedef arg< 0, data_store_field_t > p_in;
    typedef arg< 1, data_store_field_t > p_out;
    typedef arg< 2, data_store_field_t, enumtype::default_location_type, true > p_tmp;

    typedef boost::mpl::vector< p_in, p_out, p_tmp > accessor_list;
    aggregator_type< accessor_list > domain(dsf_in, dsf_out);

    int halo_size = 0;

    uint_t di[5] = {halo_size, halo_size, halo_size, d1 - 1 - halo_size, d1};
    uint_t dj[5] = {halo_size, halo_size, halo_size, d2 - 1 - halo_size, d2};

    grid< axis > gr(di, dj);
    gr.value_list[0] = 0;
    gr.value_list[1] = d3 - 1;

    auto z = make_computation< be >(domain,
        gr,
        make_multistage(execute< forward >(),
                                        define_caches(cache< IJ, local >(p_tmp())),
                                        make_stage< A >(p_in(), p_tmp()),
                                        make_stage< A >(p_tmp(), p_out())));

    z->ready();
    z->steady();
    z->run();
    z->finalize();
    std::cout << "time: " << z->print_meter() << std::endl;

    bool valid = true;
    for (int i = halo_size; i < d1 - halo_size; ++i) {
        for (int j = halo_size; j < d2 - halo_size; ++j) {
            for (int k = 0; k < d3; ++k) {
                valid &= (hv_out.template get< 0, 0 >()(i, j, k) == 1);

                valid &= (hv_out.template get< 1, 0 >()(i, j, k) == 2);
                valid &= (hv_out.template get< 1, 1 >()(i, j, k) == 3);

                valid &= (hv_out.template get< 2, 0 >()(i, j, k) == 4);
                valid &= (hv_out.template get< 2, 1 >()(i, j, k) == 5);
                valid &= (hv_out.template get< 2, 2 >()(i, j, k) == 6);

                if (!valid) {
                    std::cout << "ERROR IN: " << i << " " << j << " " << k << std::endl;
                    abort();
                }
            }
        }
    }

    std::cout << "valid: " << valid << std::endl;
    return 0;
}
