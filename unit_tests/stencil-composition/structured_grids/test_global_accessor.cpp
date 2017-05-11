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
#define PEDANTIC_DISABLED

#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <storage/storage-facility.hpp>

using namespace gridtools;
using namespace enumtype;

typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;
typedef interval< level< 0, -2 >, level< 1, 1 > > axis;
#ifdef __CUDACC__
typedef backend< Cuda, structured, Block > backend_t;
typedef storage_traits< Cuda > storage_traits_t;
#else
typedef backend< Host, structured, Naive > backend_t;
typedef storage_traits< Host > storage_traits_t;
#endif
typedef storage_traits_t::storage_info_t< 0, 3 > storage_info_t;
typedef storage_traits_t::data_store_t< float_type, storage_info_t > data_store_t;

struct boundary {

    int int_value;

    boundary() {}
    boundary(int ival) : int_value(ival) {}

    GT_FUNCTION
    double value() const { return 10.; }
};

struct functor {
    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 > > sol;
    typedef global_accessor< 1, enumtype::inout > bd;

    typedef boost::mpl::vector< sol, bd > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
        eval(sol()) += eval(bd()).value() + eval(bd()).int_value;
    }
};

TEST(test_global_accessor, boundary_conditions) {
    storage_info_t sinfo(10, 10, 10);
    data_store_t sol_(sinfo, 2.);

    boundary bd(20);

    auto bd_ = backend_t::make_global_parameter(bd);
    typedef arg< 1, decltype(bd_) > p_bd;

    halo_descriptor di = halo_descriptor(0, 1, 1, 9, 10);
    halo_descriptor dj = halo_descriptor(0, 1, 1, 1, 2);
    grid< axis > coords_bc(di, dj);
    coords_bc.value_list[0] = 0;
    coords_bc.value_list[1] = 1;

    typedef arg< 0, data_store_t > p_sol;

    aggregator_type< boost::mpl::vector< p_sol, p_bd > > domain(sol_, bd_);

    /*****RUN 1 WITH bd int_value set to 20****/
    auto bc_eval = make_computation< backend_t >(
        domain, coords_bc, make_multistage(execute< forward >(), make_stage< functor >(p_sol(), p_bd())));

    bc_eval->ready();
    bc_eval->steady();
    bc_eval->run();
    // fetch data and check
    sol_.sync();
    auto solv = make_host_view(sol_);
    bool result = true;
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 10.;
                    value += 20;
                }
                if (solv(i, j, k) != value) {
                    result = false;
                }
            }

    // get the configuration object from the gpu
    // modify configuration object (boundary)
    bd.int_value = 30;
    backend_t::update_global_parameter(bd_, bd);

    // get the storage object from the gpu
    // modify storage object
    for (unsigned i = 0; i < 10; ++i) {
        for (unsigned j = 0; j < 10; ++j) {
            for (unsigned k = 0; k < 10; ++k) {
                solv(i, j, k) = 2.;
            }
        }
    }

    sol_.sync();
    sol_.reactivate_host_write_views();

    // run again and finalize
    bc_eval->run();
    bc_eval->finalize();

    // check result of second run
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 10.;
                    value += 30;
                }
                if (solv(i, j, k) != value) {
                    result = false;
                }
            }

    EXPECT_TRUE(result);
}
