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

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;

typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > kfull;

typedef gridtools::interval< level< 0, 1 >, level< 1, -1 > > kbody_high;
typedef gridtools::interval< level< 0, -1 >, level< 0, -1 > > kminimum;
typedef gridtools::interval< level< 1, -1 >, level< 1, -1 > > kmaximum;
typedef gridtools::interval< level< 0, -1 >, level< 1, -2 > > kbody_low;

// These are the stencil operators that compose the multistage stencil in this test
struct shif_acc_forward {

    typedef accessor< 0, enumtype::in, extent<> > in;
    typedef accessor< 1, enumtype::inout, extent<> > out;
    typedef accessor< 2, enumtype::inout, extent< 0, 0, 0, 0, -1, 0 > > buff;

    typedef boost::mpl::vector< in, out, buff > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimum) {
        eval(buff()) = eval(in());
        eval(out()) = eval(buff());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_high) {

        eval(buff()) = eval(buff(0, 0, -1)) + eval(in());
        eval(out()) = eval(buff());
    }
};

struct shif_acc_backward {

    typedef accessor< 0, enumtype::in, extent<> > in;
    typedef accessor< 1, enumtype::inout, extent<> > out;
    typedef accessor< 2, enumtype::inout, extent< 0, 0, 0, 0, 0, 1 > > buff;

    typedef boost::mpl::vector< in, out, buff > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximum) {
        eval(buff()) = eval(in());
        eval(out()) = eval(buff());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_low) {
        eval(buff()) = eval(buff(0, 0, 1)) + eval(in());
        eval(out()) = eval(buff());
    }
};

/*
 * The following operators and structs are for debugging only
 */
std::ostream &operator<<(std::ostream &s, shif_acc_forward const) { return s << "shif_acc_forward"; }

TEST(kcache, local_forward) {

    uint_t d1 = 6;
    uint_t d2 = 6;
    uint_t d3 = 10;

#ifdef __CUDACC__
#define BACKEND_ARCH Cuda
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND_ARCH Host
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    typedef storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 > storage_info_t;
    typedef storage_traits< BACKEND_ARCH >::data_store_t< float_type, storage_info_t > data_store_t;

    storage_info_t meta_data_(d1, d2, d3);

    // Definition of the actual data fields that are used for input/output
    data_store_t in(meta_data_);
    data_store_t ref(meta_data_);
    data_store_t out(meta_data_);

    in.allocate();
    out.allocate();
    ref.allocate();

    auto in_v = make_host_view(in);
    auto out_v = make_host_view(out);
    auto ref_v = make_host_view(ref);

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            in_v(i, j, 0) = i + j;
            ref_v(i, j, 0) = in_v(i, j, 0);
            for (uint_t k = 1; k < d3; ++k) {
                in_v(i, j, k) = i + j + k;
                ref_v(i, j, k) = ref_v(i, j, k - 1) + in_v(i, j, k);
                out_v(i, j, k) = -1;
            }
        }
    }

    typedef arg< 0, data_store_t > p_in;
    typedef arg< 1, data_store_t > p_out;
    typedef tmp_arg< 2, data_store_t > p_buff;

    typedef boost::mpl::vector< p_in, p_out, p_buff > accessor_list;
    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
    // that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in
    // order. (I don't particularly like this)
    gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out));

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
            define_caches(cache< K, local, kfull >(p_buff())),
            gridtools::make_stage< shif_acc_forward >(p_in() // esf_descriptor
                ,
                p_out(),
                p_buff())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

    out.sync();
    out.reactivate_host_write_views();

    bool success = true;
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (ref_v(i, j, k) != out_v(i, j, k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "ref = " << ref_v(i, j, k) << ", out = " << out_v(i, j, k) << std::endl;
                    success = false;
                }
            }
        }
    }
    kcache_stencil->finalize();

    ASSERT_TRUE(success);
}

TEST(kcache, local_backward) {

    uint_t d1 = 6;
    uint_t d2 = 6;
    uint_t d3 = 10;

#ifdef __CUDACC__
#define BACKEND_ARCH Cuda
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND_ARCH Host
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    typedef storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 > storage_info_t;
    typedef storage_traits< BACKEND_ARCH >::data_store_t< float_type, storage_info_t > data_store_t;

    storage_info_t meta_data_(d1, d2, d3);

    // Definition of the actual data fields that are used for input/output
    data_store_t in(meta_data_);
    data_store_t ref(meta_data_);
    data_store_t out(meta_data_);

    in.allocate();
    out.allocate();
    ref.allocate();

    auto in_v = make_host_view(in);
    auto out_v = make_host_view(out);
    auto ref_v = make_host_view(ref);

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            in_v(i, j, d3 - 1) = i + j + d3 - 1;
            ref_v(i, j, d3 - 1) = in_v(i, j, d3 - 1);
            for (int_t k = d3 - 2; k >= 0; --k) {
                in_v(i, j, k) = i + j + k;
                ref_v(i, j, k) = ref_v(i, j, k + 1) + in_v(i, j, k);
            }
        }
    }

    typedef arg< 0, data_store_t > p_in;
    typedef arg< 1, data_store_t > p_out;
    typedef tmp_arg< 2, data_store_t > p_buff;

    typedef boost::mpl::vector< p_in, p_out, p_buff > accessor_list;
    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
    // that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in
    // order. (I don't particularly like this)
    gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out));

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
        (execute< backward >(),
            define_caches(cache< K, local, kfull >(p_buff())),
            gridtools::make_stage< shif_acc_backward >(p_in() // esf_descriptor
                ,
                p_out(),
                p_buff())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

    out.sync();
    out.reactivate_host_write_views();

    bool success = true;
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (ref_v(i, j, k) != out_v(i, j, k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "ref = " << ref_v(i, j, k) << ", out = " << out_v(i, j, k) << std::endl;
                    success = false;
                }
            }
        }
    }
    kcache_stencil->finalize();

    ASSERT_TRUE(success);
}
