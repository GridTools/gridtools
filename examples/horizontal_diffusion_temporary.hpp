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
#pragma once

#include "interface1.hpp"

/**
  @file
  This file implements an acceptance test for temporary storages:
  it mimics the case of the horizontal diffusion example, without using caches
*/

namespace horizontal_diffusion_temporary {

    using namespace horizontal_diffusion;

    struct copy_functor {

        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_lap) {
            eval(out()) = eval(in());
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        typedef layout_ijk layout_t;

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_data_t;
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_type;
        typedef BACKEND::temporary_storage_type< float_type, meta_data_t >::type tmp_storage_type;

        meta_data_t meta_data_(x, y, z);

        // Definition of the actual data fields that are used for input/output
        storage_type in(meta_data_, "in");
        storage_type out(meta_data_, float_type(-1.), "out");
        storage_type coeff(meta_data_, float_type(0.025), "coeff");

        // const double PI = std::atan(1.) * 4.;

        // double dx = 1. / (double)(d1);
        // double dy = 1. / (double)(d2);

        //     double delta0 = (0.995156 - 0.994954) / (double)(d2 - 1);
        //     double delta1 = (0.995143 - 0.994924) / (double)(d2 - 1);

        //     for (uint_t j = 0; j < d2; j++) {
        //         for (uint_t i = 0; i < d1; i++) {
        //             double x = dx * i;
        //             double y = dy * j;
        //             for (uint_t k = 0; k < d3; k++) {
        //                 // in values between 5 and 9
        //                 in(i, j, k) = 5. + 8 * (2. + cos(PI * (x + 1.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;
        //             }
        //         }
        //     }

        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    in(i, j, k) =
                        i + j * 100 + k * 10000; // cannot chose a linear functional because its laplacian is 0
                }

        typedef arg< 0, storage_type > p_in;
        typedef arg< 1, storage_type > p_out;
        typedef arg< 2, tmp_storage_type > p_lap;
        typedef arg< 3, tmp_storage_type > p_flx;
        typedef arg< 4, tmp_storage_type > p_fly;
        typedef arg< 5, storage_type > p_coeff;

        typedef boost::mpl::vector< p_in, p_out, p_lap, p_flx, p_fly, p_coeff > accessor_list;
        gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out, &coeff));

        uint_t di[5] = {2, 2, 2, d1 - 3, d1};
        uint_t dj[5] = {2, 2, 2, d2 - 3, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::stencil *
#else
        boost::shared_ptr< gridtools::stencil >
#endif
#endif
            copy = gridtools::make_computation< gridtools::BACKEND >(
                domain,
                grid,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< lap_function >(p_lap(), p_in()),
                    // gridtools::make_stage< copy_functor >(p_lap(),p_out())
                    gridtools::make_independent(gridtools::make_stage< flx_function >(p_flx(), p_in(), p_lap()),
                        gridtools::make_stage< fly_function >(p_fly(), p_in(), p_lap())),
                    gridtools::make_stage< out_function >(p_out(), p_in(), p_flx(), p_fly(), p_coeff())));

        copy->ready();

        copy->steady();

        copy->run();

#ifdef __CUDACC__
        out.d2h_update();
        in.d2h_update();
#endif

        //////////// TEST VERIFICATION ////////////
        storage_type lap_ref(meta_data_, "lap");
        storage_type flx_ref(meta_data_, "flx");
        storage_type fly_ref(meta_data_, "fly");
        storage_type out_ref(meta_data_, "fly");
        storage_type coeff_ref(meta_data_, float_type(1.), "coeff");

        double epsilon = 1e-10;
        bool success = true;
        if (verify) {
            for (uint_t i = 1; i < d1 - 1; ++i) {
                for (uint_t j = 1; j < d2 - 1; ++j) {
                    for (uint_t k = 0; k < d3; ++k) {

                        lap_ref(i, j, k) = 4. * in(i, j, k) -
                                           (in(i - 1, j, k) + in(i, j - 1, k) + in(i + 1, j, k) + (in(i, j + 1, k)));
                    }
                }
            }

            // for (uint_t i = 1; i < d1-1; ++i) {
            //         for (uint_t j = 1; j < d2-1; ++j) {
            //             for (uint_t k = 0; k < d3; ++k) {
            //                 if (lap_ref(i,j,k) <= out(i, j, k)-epsilon
            //                     ||
            //                     lap_ref(i,j,k) >= out(i, j, k)+epsilon ) {
            //                     std::cout << "error in " << i << ", " << j << ", " << k << ": "
            //                               << "in = " << lap_ref(i,j,k) << ", out = " << out(i, j, k) << std::endl;
            //                     success = false;
            //                 }
            //             }
            //         }
            //     }
            // }

            for (uint_t i = 1; i < d1 - 2; ++i) {
                for (uint_t j = 1; j < d2 - 1; ++j) {
                    for (uint_t k = 0; k < d3; ++k) {

                        flx_ref(i, j, k) = lap_ref(i + 1, j, k) - lap_ref(i, j, k);
                        if (flx_ref(i, j, k) * (in(i + 1, j, k) - in(i, j, k)) > 0) {
                            flx_ref(i, j, k) = 0.;
                        }
                    }
                }
            }

            for (uint_t i = 1; i < d1 - 2; ++i) {
                for (uint_t j = 1; j < d2 - 2; ++j) {
                    for (uint_t k = 0; k < d3; ++k) {

                        fly_ref(i, j, k) = lap_ref(i, j + 1, k) - lap_ref(i, j, k);
                        if (fly_ref(i, j, k) * (in(i, j + 1, k) - in(i, j, k)) > 0) {
                            fly_ref(i, j, k) = 0.;
                        }
                    }
                }
            }

            for (uint_t i = 2; i < d1 - 2; ++i) {
                for (uint_t j = 2; j < d2 - 2; ++j) {
                    for (uint_t k = 0; k < d3; ++k) {

                        out_ref(i, j, k) = in(i, j, k) -
                                           coeff_ref(i, j, k) * (flx_ref(i, j, k) - flx_ref(i - 1, j, k) +
                                                                    fly_ref(i, j, k) - fly_ref(i, j - 1, k));
                    }
                }
            }

            for (uint_t i = 2; i < d1 - 2; ++i) {
                for (uint_t j = 2; j < d2 - 2; ++j) {
                    for (uint_t k = 0; k < d3; ++k) {
                        if (out_ref(i, j, k) <= out(i, j, k) - epsilon || out_ref(i, j, k) >= out(i, j, k) + epsilon) {
                            std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                      << "in = " << out_ref(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                            success = false;
                        }
                    }
                }
            }
        }

        // #ifdef BENCHMARK
        //         benchmarker::run(copy, t_steps);
        // #endif
        copy->finalize();

        return success;
    }
} // namespace copy_stencil
