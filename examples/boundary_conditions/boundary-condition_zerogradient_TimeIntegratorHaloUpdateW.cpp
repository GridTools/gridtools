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
#include <gridtools.hpp>
#include <common/halo_descriptor.hpp>

#include <boundary-conditions/boundary.hpp>
#include <boundary-conditions/direction.hpp>

using gridtools::direction;
using gridtools::sign;
using gridtools::minus_;
using gridtools::zero_;
using gridtools::plus_;

#include <stencil-composition/stencil-composition.hpp>

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>

#include "../benchmarker.hpp"

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define GT_ARCH Cuda
#else
#define GT_ARCH Host
#endif

#define BACKEND backend< GT_ARCH, GRIDBACKEND, Block >
struct zero_grad_bc {
    uint_t first_i, last_i, first_j, last_j;

    GT_FUNCTION zero_grad_bc(uint_t first_i, uint_t last_i, uint_t first_j, uint_t last_j)
        : first_i(first_i), last_i(last_i), first_j(first_j), last_j(last_j) {}

    // relative coordinates
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< minus_, minus_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {
        data_field(i, j, k) = data_field(first_i, first_j, k);
    }
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< minus_, zero_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {
        data_field(i, j, k) = data_field(first_i, j, k);
    }
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< minus_, plus_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {
        data_field(i, j, k) = data_field(first_i, last_j, k);
    }
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< zero_, minus_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {
        data_field(i, j, k) = data_field(i, first_j, k);
    }
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< zero_, zero_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {}
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< zero_, plus_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {
        data_field(i, j, k) = data_field(i, last_j, k);
    }
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< plus_, minus_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {
        data_field(i, j, k) = data_field(last_i, first_j, k);
    }
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< plus_, zero_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {
        data_field(i, j, k) = data_field(last_i, j, k);
    }
    template < sign K, typename DataField >
    GT_FUNCTION void operator()(
        gridtools::direction< plus_, plus_, K >, DataField &data_field, uint_t i, uint_t j, uint_t k) const {
        data_field(i, j, k) = data_field(last_i, last_j, k);
    }
};

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " dimx dimy dimz [t_steps]\n"
                                             " where args are integer sizes of the data fields"
                  << std::endl;
        return EXIT_FAILURE;
    }

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);
    uint_t t_steps = (argc > 4) ? atoi(argv[4]) : 1;

    typedef BACKEND::storage_traits_t::storage_info_t< 0, 3, halo< 1, 1, 1 > > meta_data_t;
    typedef BACKEND::storage_traits_t::data_store_t< int_t, meta_data_t > storage_t;

    // Definition of the actual data fields that are used for input/output
    meta_data_t meta_(d1, d2, d3);
    storage_t out(meta_,
        [d1, d2](int i, int j, int k) {
            if (i < 3 || i > d1 - 4 || j < 3 || j > d2 - 4)
                return -1.;
            else if (i == 3 && j > 3 && j < d2 - 4)
                return 1.;
            else if (i == d1 - 4 && j > 3 && j < d2 - 4)
                return 3.;
            else if (j == 3 && i > 3 && i < d1 - 4)
                return 2.;
            else if (j == d2 - 4 && i > 3 && i < d1 - 4)
                return 4.;
            else if (i == 3 && j == 3)
                return 5.;
            else if (i == 3 && j == d2 - 4)
                return 6.;
            else if (i == d1 - 4 && j == 3)
                return 7.;
            else if (i == d1 - 4 && j == d2 - 4)
                return 8.;
            else
                return 555.;
        },
        "out");

    storage_t ref(meta_,
        [d1, d2](int i, int j, int k) {
            if (i <= 3 && j <= 3)
                return 5.;
            else if (i <= 3 && j >= d2 - 4)
                return 6.;
            else if (i >= d1 - 4 && j <= 3)
                return 7.;
            else if (i >= d1 - 4 && j >= d2 - 4)
                return 8.;
            else if (i <= 3 && (j > 3 && j < d2 - 4))
                return 1.;
            else if (i >= d1 - 4 && (j > 3 && j < d2 - 4))
                return 3.;
            else if (j <= 3 && (i > 3 && i < d1 - 4))
                return 2.;
            else if (j >= d2 - 4 && (i > 3 && i < d1 - 4))
                return 4.;
            else
                return 555.;
        },
        "ref");

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(3, 3, 3, d1 - 4, d1);
    halos[1] = gridtools::halo_descriptor(3, 3, 3, d2 - 4, d2);
    halos[2] = gridtools::halo_descriptor(0, 0, 0, d3 - 1, d3);

    // sync the data stores if needed
    out.sync();

    gridtools::boundary< zero_grad_bc, GT_ARCH > bc(
        halos, zero_grad_bc(halos[0].begin(), halos[0].end(), halos[1].begin(), halos[1].end()));
    bc.apply(out);

    // sync the data stores if needed
    out.sync();

    bool success = true;
    bool verify = true;
    auto out_v = make_host_view(out);
    auto ref_v = make_host_view(ref);

    if (verify) {
        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    if (out_v(i, j, k) != ref_v(i, j, k)) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "out = " << out_v(i, j, k) << ", ref = " << ref_v(i, j, k) << std::endl;
                        success = false;
                    }
                }
            }
        }
    }
    if (!success)
        exit(1);

#ifdef BENCHMARK
    out.sync();
    benchmarker::run_bc(bc, t_steps, out);
#endif

    //    return error;
}
