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

template < typename T >
struct interpolation_bc {
    T weight1, weight2;

    GT_FUNCTION
    interpolation_bc(T weight1, T weight2) : weight1(weight1), weight2(weight2) {}

    // relative coordinates
    template < typename Direction, typename DataField >
    GT_FUNCTION void operator()(Direction,
        DataField &data_field0,
        DataField const &data_field1,
        DataField const &data_field2,
        uint_t i,
        uint_t j,
        uint_t k) const {
        data_field0(i, j, k) = data_field1(i, j, k) * weight1 + data_field2(i, j, k) * weight2;
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
    storage_t in1(meta_,
        [d1, d2](int i, int j, int k) {
            if (i < 3 || i > d1 - 4 || j < 3 || j > d2 - 4)
                return 1.;
            else
                return -1.;
        },
        "in1");
    storage_t in2(meta_,
        [d1, d2](int i, int j, int k) {
            if (i < 3 || i > d1 - 4 || j < 3 || j > d2 - 4)
                return 2.;
            else
                return -1.;
        },
        "in2");
    storage_t out(meta_, 0, "out");

    storage_t ref(meta_,
        [d1, d2](int i, int j, int k) {
            if ((i == 2 && j >= 2 && j <= d2 - 3) || (i == d1 - 3 && j >= 2 && j <= d2 - 3) ||
                (j == 2 && i > 2 && i <= d1 - 3) || (j == d2 - 3 && i > 2 && i <= d1 - 3))
                return 1.5;
            else
                return 0.;
        },
        "ref");

    gridtools::array< gridtools::halo_descriptor, 3 > halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 3, d1 - 4, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 3, d2 - 4, d2);
    halos[2] = gridtools::halo_descriptor(0, 0, 0, d3 - 1, d3);

    // sync the data stores if needed
    in1.sync();
    in2.sync();
    out.sync();

    gridtools::boundary< interpolation_bc< float_type >, GT_ARCH > bc(halos, interpolation_bc< float_type >(0.5, 0.5));
    bc.apply(out, in1, in2);

    // sync the data stores if needed
    in1.sync();
    in2.sync();
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

// making the views to access and check correctness
//    auto inv1 = make_host_view(in1);
//    auto inv2 = make_host_view(in2);
//    auto outv = make_host_view(outv);
//
//    assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
//    assert(check_consistency(out, out) && "view is in an inconsistent state.");
//
//    // reactivate views and check consistency
//    in_s.reactivate_host_write_views();
//    out.reactivate_host_write_views();
//    assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
//    assert(check_consistency(out, out) && "view is in an inconsistent state.");

//    if (false) {
//        bool error = false;
//        // check inner domain (should be zero)
//        for (uint_t i = 1; i < d3 - 1; ++i) {
//            for (uint_t j = 1; j < d2 - 1; ++j) {
//                for (uint_t k = 1; k < d1 - 1; ++k) {
//                    if (in(k, j, i) != i + j + k) {
//                        std::cout << "Error: INPUT field got modified " << k << " " << j << " " << i << "\n";
//                        error = true;
//                    }
//                    if (out(k, j, i) != 0) {
//                        std::cout << "Error: Inner domain of OUTPUT field got modified " << k << " " << j << " " << i
//                                  << "\n";
//                        error = true;
//                    }
//                }
//            }
//        }
//
//        // check edge column
//        if (out(0, 0, 0) != 111110) {
//            std::cout << "Error: edge column values in OUTPUT field are wrong 0 0 0\n";
//            error = true;
//        }
//        for (uint_t k = 1; k < d3; ++k) {
//            if (out(0, 0, k) != 155554) {
//                std::cout << "Error: edge column values in OUTPUT field are wrong 0 0 " << k << "\n";
//                error = true;
//            }
//        }
//
//        // check j==0 i>0 surface
//        for (uint_t i = 1; i < d1; ++i) {
//            for (uint_t k = 0; k < d3; ++k) {
//                if (out(i, 0, k) != 176) {
//                    std::cout << "Error: j==0 surface values in OUTPUT field are wrong " << i << " 0 " << k << "\n";
//                    error = true;
//                }
//            }
//        }
//
//        // check outer domain
//        for (uint_t i = 0; i < d1; ++i) {
//            for (uint_t j = 0; j < d2; ++j) {
//                for (uint_t k = 0; k < d3; ++k) {
//                    // check outer surfaces of the cube
//                    if (((i == 0 || i == d1 - 1) && j > 0) || (j > 0 && (k == 0 || k == d3 - 1))) {
//                        if (out(i, j, k) != in(i, j, k) * 2) {
//                            std::cout << "Error: values in OUTPUT field are wrong " << i << " " << j << " " << k
//                                      << "\n";
//                            error = true;
//                        }
//                    }
//                }
//            }
//        }
//
//        if (error) {
//            std::cout << "TEST failed.\n";
//            abort();
//        }
//    }

#ifdef BENCHMARK
    in1.sync();
    in2.sync();
    out.sync();
    benchmarker::run_bc(bc, t_steps, in1, in2, out);
#endif

    //    return error;
}
