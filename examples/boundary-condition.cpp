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

#ifdef CUDA_EXAMPLE
#include <boundary-conditions/apply_gpu.hpp>
#else
#include <boundary-conditions/apply.hpp>
#endif

using gridtools::direction;
using gridtools::sign;
using gridtools::minus_;
using gridtools::zero_;
using gridtools::plus_;

#include <stencil-composition/stencil-composition.hpp>

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

template < typename T >
struct direction_bc_input {
    T value;

    GT_FUNCTION
    direction_bc_input() : value(1) {}

    GT_FUNCTION
    direction_bc_input(T v) : value(v) {}

    // relative coordinates
    template < typename Direction, typename DataField0, typename DataField1 >
    GT_FUNCTION void operator()(
        Direction, DataField0 &data_field0, DataField1 const &data_field1, uint_t i, uint_t j, uint_t k) const {
        data_field1(i, j, k) = data_field0(i, j, k) * value;
    }

    // relative coordinates
    template < sign I, sign K, typename DataField0, typename DataField1 >
    GT_FUNCTION void operator()(direction< I, minus_, K >,
        DataField0 &data_field0,
        DataField1 const &data_field1,
        uint_t i,
        uint_t j,
        uint_t k) const {
        data_field1(i, j, k) = 88 * value;
    }

    // relative coordinates
    template < sign K, typename DataField0, typename DataField1 >
    GT_FUNCTION void operator()(direction< minus_, minus_, K >,
        DataField0 &data_field0,
        DataField1 const &data_field1,
        uint_t i,
        uint_t j,
        uint_t k) const {
        data_field1(i, j, k) = 77777 * value;
    }

    template < typename DataField0, typename DataField1 >
    GT_FUNCTION void operator()(direction< minus_, minus_, minus_ >,
        DataField0 &data_field0,
        DataField1 const &data_field1,
        uint_t i,
        uint_t j,
        uint_t k) const {
        data_field1(i, j, k) = 55555 * value;
    }
};

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " dimx dimy dimz\n"
                                             " where args are integer sizes of the data fields"
                  << std::endl;
        return EXIT_FAILURE;
    }

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);

    typedef BACKEND::storage_traits_t::storage_info_t< 0, 3, halo< 1, 1, 1 > > meta_data_t;
    typedef BACKEND::storage_traits_t::data_store_t< int_t, meta_data_t > storage_t;

    // Definition of the actual data fields that are used for input/output
    meta_data_t meta_(d1 - 2, d2 - 2, d3 - 2);
    storage_t in_s(meta_, [](int i, int j, int k) { return i + j + k; }, "in");
    storage_t out_s(meta_, 0, "out");

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    auto in = make_host_view(in_s);
    auto out = make_host_view(out_s);
    // sync the data stores if needed
    in_s.sync();
    out_s.sync();

#ifdef __CUDACC__
    auto dvin = make_device_view(in_s);
    auto dvout = make_device_view(out_s);

    gridtools::boundary_apply_gpu< direction_bc_input< uint_t > >(halos, direction_bc_input< uint_t >(2))
        .apply(dvin, dvout);
#else

    gridtools::boundary_apply< direction_bc_input< uint_t > >(halos, direction_bc_input< uint_t >(2)).apply(in, out);
#endif

    // sync the data stores if needed
    in_s.sync();
    out_s.sync();

    // reactivate views and check consistency
    in_s.reactivate_host_write_views();
    out_s.reactivate_host_write_views();
    assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
    assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

    // check inner domain (should be zero)
    bool error = false;
    for (uint_t i = 1; i < d3 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d1 - 1; ++k) {
                if (in(k, j, i) != i + j + k) {
                    std::cout << "Error: INPUT field got modified " << k << " " << j << " " << i << "\n";
                    error = true;
                }
                if (out(k, j, i) != 0) {
                    std::cout << "Error: Inner domain of OUTPUT field got modified " << k << " " << j << " " << i
                              << "\n";
                    error = true;
                }
            }
        }
    }

    // check edge column
    if (out(0, 0, 0) != 111110) {
        std::cout << "Error: edge column values in OUTPUT field are wrong 0 0 0\n";
        error = true;
    }
    for (uint_t k = 1; k < d3; ++k) {
        if (out(0, 0, k) != 155554) {
            std::cout << "Error: edge column values in OUTPUT field are wrong 0 0 " << k << "\n";
            error = true;
        }
    }

    // check j==0 i>0 surface
    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t k = 0; k < d3; ++k) {
            if (out(i, 0, k) != 176) {
                std::cout << "Error: j==0 surface values in OUTPUT field are wrong " << i << " 0 " << k << "\n";
                error = true;
            }
        }
    }

    // check outer domain
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                // check outer surfaces of the cube
                if (((i == 0 || i == d1 - 1) && j > 0) || (j > 0 && (k == 0 || k == d3 - 1))) {
                    if (out(i, j, k) != in(i, j, k) * 2) {
                        std::cout << "Error: values in OUTPUT field are wrong " << i << " " << j << " " << k << "\n";
                        error = true;
                    }
                }
            }
        }
    }

    if (error) {
        std::cout << "TEST failed.\n";
        abort();
    }

    return error;
}
