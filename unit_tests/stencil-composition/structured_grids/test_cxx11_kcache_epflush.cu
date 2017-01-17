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
#include "gtest/gtest.h"
#include "stencil-composition/stencil-composition.hpp"

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
//                   strides  1 x xy
//                      dims  x y z
typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#endif

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;

typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > kfull;

typedef gridtools::interval< level< 0, 2 >, level< 1, -1 > > kbody_high;
typedef gridtools::interval< level< 0, -1 >, level< 0, 1 > > kminimum;

typedef gridtools::interval< level< 0, -1 >, level< 2, 1 > > axis_b;
typedef gridtools::interval< level< 1, 1 >, level< 2, -1 > > kmaximum_b;
typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > kbody_low_b;
typedef gridtools::interval< level< 0, -1 >, level< 2, -1 > > kfull_b;

// These are the stencil operators that compose the multistage stencil in this test
struct shift_acc_forward {

    typedef accessor< 0, enumtype::in, extent<> > in;
    typedef accessor< 1, enumtype::inout, extent< 0, 0, 0, 0, -2, 0 > > out;

    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimum) {
        eval(out()) = eval(in());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_high) {
        eval(out()) = eval(out(0, 0, -1)) + eval(out(0, 0, -2)) + eval(in());
    }
};

struct shift_acc_backward {

    typedef accessor< 0, enumtype::in, extent<> > in;
    typedef accessor< 1, enumtype::inout, extent< 0, 0, 0, 0, 0, 2 > > out;

    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximum_b) {
        eval(out()) = eval(in());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_low_b) {
        eval(out()) = eval(out(0, 0, 1)) + eval(out(0, 0, 2)) + eval(in());
    }
};

/*
 * The following operators and structs are for debugging only
 */
std::ostream &operator<<(std::ostream &s, shift_acc_forward const) { return s << "shift_acc_forward"; }

TEST(kcache, epflush_forward) {

    uint_t d1 = 6;
    uint_t d2 = 6;
    uint_t d3 = 10;

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
    typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_t;
    typedef BACKEND::temporary_storage_type< float_type, meta_data_t >::type tmp_storage_t;

    meta_data_t meta_data_(d1, d2, d3);

    // Definition of the actual data fields that are used for input/output
    typedef storage_t storage_type;
    storage_type in(meta_data_, "in");
    storage_type ref(meta_data_, "ref");

    storage_type out(meta_data_, float_type(-1.));
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                in(i, j, k) = i + j + k;
            }

            ref(i, j, 0) = in(i, j, 0);
            ref(i, j, 1) = in(i, j, 1);

            for (uint_t k = 2; k < d3; ++k) {
                ref(i, j, k) = ref(i, j, k - 1) + ref(i, j, k - 2) + in(i, j, k);
            }
        }
    }

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_type > p_out;

    typedef boost::mpl::vector< p_in, p_out > accessor_list;
    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
    // that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in
    // order. (I don't particularly like this)
    gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
    uint_t di[5] = {0, 0, 0, d1 - 1, d1};
    uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    auto kcache_stencil = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage // mss_descriptor
        (execute< forward >(),
            define_caches(cache< K, epflush, kfull >(p_out())),
            gridtools::make_stage< shift_acc_forward >(p_in() // esf_descriptor
                ,
                p_out())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

#ifdef __CUDACC__
    out.d2h_update();
    in.d2h_update();
#endif

    bool success = true;
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3 - 2; ++k) {
                if (ref(i, j, k) == out(i, j, k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "ref = " << ref(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                    success = false;
                }
            }
            for (uint_t k = d3 - 2; k < d3; ++k) {
                if (ref(i, j, k) != out(i, j, k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "ref = " << ref(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                    success = false;
                }
            }
        }
    }
    kcache_stencil->finalize();

    ASSERT_TRUE(success);
}

TEST(kcache, epflush_backward) {

    uint_t d1 = 6;
    uint_t d2 = 6;
    uint_t d3 = 10;

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
    typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_t;
    typedef BACKEND::temporary_storage_type< float_type, meta_data_t >::type tmp_storage_t;

    meta_data_t meta_data_(d1, d2, d3);

    // Definition of the actual data fields that are used for input/output
    typedef storage_t storage_type;
    storage_type in(meta_data_, "in");
    storage_type ref(meta_data_, "ref");

    storage_type out(meta_data_, float_type(-1.));
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (int_t k = 0; k < d3; ++k) {
                in(i, j, k) = i + j + k;
            }

            ref(i, j, d3 - 1) = in(i, j, d3 - 1);
            ref(i, j, d3 - 2) = in(i, j, d3 - 2);

            for (int_t k = d3 - 3; k >= 0; --k) {
                ref(i, j, k) = ref(i, j, k + 1) + ref(i, j, k + 2) + in(i, j, k);
            }
        }
    }

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_type > p_out;

    typedef boost::mpl::vector< p_in, p_out > accessor_list;
    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
    // that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in
    // order. (I don't particularly like this)
    gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
    uint_t di[5] = {0, 0, 0, d1 - 1, d1};
    uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

    gridtools::grid< axis_b > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 3;
    grid.value_list[2] = d3 - 1;

    auto kcache_stencil = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage // mss_descriptor
        (execute< forward >(),
            define_caches(cache< K, epflush, kfull_b >(p_out())),
            gridtools::make_stage< shift_acc_backward >(p_in() // esf_descriptor
                ,
                p_out())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

#ifdef __CUDACC__
    out.d2h_update();
    in.d2h_update();
#endif

    bool success = true;
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 2; ++k) {
                if (ref(i, j, k) == out(i, j, k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "ref = " << ref(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                    success = false;
                }
            }
            for (uint_t k = 2; k < d3; ++k) {
                if (ref(i, j, k) != out(i, j, k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "ref = " << ref(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                    success = false;
                }
            }
        }
    }
    kcache_stencil->finalize();

    ASSERT_TRUE(success);
}
