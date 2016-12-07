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

#include <iostream>

#include <stencil-composition/aggregator_type.hpp>
#include <stencil-composition/arg.hpp>
#include <stencil-composition/backend.hpp>
#include <stencil-composition/make_stage.hpp>
#include <stencil-composition/make_stencils_cxx11.hpp>
#include <storage-facility.hpp>
#define POSITIONAL_WHEN_DEBUGGING true
#include <stencil-composition/grid.hpp>
#include <stencil-composition/make_computation_cxx11.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND_V Cuda
typedef backend< BACKEND_V, GRIDBACKEND, Block > be;
#else
#define BACKEND_V Host
typedef backend< BACKEND_V, GRIDBACKEND, Naive > be;
#endif

typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;
typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;

struct A {

    typedef accessor< 0, in, extent<>, 3 > pin;
    typedef accessor< 1, inout, extent<>, 3 > pout;
    typedef boost::mpl::vector< pin, pout > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        eval(pout()) = tan(pow(eval(pin()), ((int)eval(pin()) % 13)));
    }
};
struct B {

    typedef accessor< 0, in, extent<>, 3 > pin;
    typedef accessor< 1, inout, extent<>, 3 > pout;
    typedef boost::mpl::vector< pin, pout > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        eval(pout()) = eval(pin()) + 5;
    }
};

int main() {
    int d1 = 128;
    int d2 = 128;
    int d3 = 80;

    typedef storage_traits< BACKEND_V >::storage_info_t< 0, 3 > storage_info_ty;
    typedef storage_traits< BACKEND_V >::storage_info_t< 1, 3 > storage_info_ty1;
    typedef storage_traits< BACKEND_V >::data_store_t< float, storage_info_ty > data_store_t;
    typedef storage_traits< BACKEND_V >::data_store_t< float, storage_info_ty1 > data_store_t1;
    typedef storage_traits< BACKEND_V >::data_store_t< float, storage_info_ty > data_store_t2;
    typedef storage_traits< BACKEND_V >::data_store_field_t< float, storage_info_ty1, 1, 2, 3 > data_store_field_t;

    storage_info_ty si(d1, d2, d3);
    storage_info_ty1 si1(d1, d2, d3);
    data_store_field_t dsf_in(si1);
    data_store_field_t dsf_out(si1);

    dsf_in.allocate();
    dsf_out.allocate();

    data_store_t ds_in(si);
    data_store_t2 ds_w(si);
    data_store_t ds_out(si);
    ds_in.allocate();
    ds_out.allocate();
    ds_w.allocate();
    auto hv_in = make_host_view(ds_in);
    auto hv_out = make_host_view(ds_out);

    // fill with values
    unsigned x = 0;
    for (int i = 0; i < d1; ++i)
        for (int j = 0; j < d2; ++j)
            for (int k = 0; k < d3; ++k) {
                hv_in(i, j, k) = x++;
                hv_out(i, j, k) = 123;
            }

    // create some gridtools stuff
    typedef arg< 0, data_store_t, false > p_in;
    typedef arg< 1, data_store_t, false > p_out;
    typedef arg< 2, data_store_t, true > p_tmp;

    /*
    typedef arg< 4, data_store_t2, true > p_tmp;
    typedef arg< 5, data_store_t2, true > p_tmp2;
    typedef arg< 1, data_store_t2, false > p_w;*/
    typedef arg< 3, data_store_field_t, false > p_dsf_in;
    typedef arg< 4, data_store_field_t, false > p_dsf_out;

    typedef boost::mpl::vector< p_in, p_out, p_tmp /* , p_w, p_tmp2, p_dsf*/ > accessor_list;
    aggregator_type< accessor_list > domain(ds_in, ds_out);
    domain.print();

    uint_t di[5] = {0, 0, 0, d1 - 1, d1};
    uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

    grid< axis > gr(di, dj);
    gr.value_list[0] = 0;
    gr.value_list[1] = d3 - 1;
    std::cout << "###BEFORE COMPUTATION####\n";

    auto z = make_computation< be >(domain,
        gr,
        make_multistage(execute< forward >(), make_stage< B >(p_in(), p_tmp()), make_stage< B >(p_tmp(), p_out())));

    z->ready();
    domain.print();
    z->steady();

    double start = omp_get_wtime();
    z->run();
    double end = omp_get_wtime();
    std::cout << (end - start) << "s" << std::endl;

    z->finalize();
    std::cout << "time: " << z->print_meter() << std::endl;

    bool valid = true;
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                valid &= (hv_out(i, j, k) == hv_in(i, j, k) + 10); // (abs(hv_out(i, j, k) - tan(pow(hv_in(i, j, k),
                                                                   // ((int)hv_in(i, j, k) % 13)))+5) < 1e-5);
                if (!valid) {
                    std::cout << i << " " << j << " " << k << std::endl;
                    // std::cout << (tan(pow(hv_in(i, j, k), ((int)hv_in(i, j, k) % 13)))+5) << std::endl;
                    std::cout << hv_in(i, j, k) << std::endl;
                    std::cout << hv_out(i, j, k) << std::endl;
                    abort();
                }
            }
            // std::cout << "\n";
        }
        // std::cout << "\n";
    }

    std::cout << "valid: " << valid << std::endl;
    return 0;
}
