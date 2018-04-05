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
#include "kcache_fixture.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "tools/verifier.hpp"
#include "gtest/gtest.h"

using namespace gridtools;
using namespace enumtype;

// These are the stencil operators that compose the multistage stencil in this test
struct shift_acc_forward_fill_and_flush {

    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0, -1, 0 > > in;

    typedef boost::mpl::vector< in > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_high) {
        eval(in()) = eval(in()) + eval(in(0, 0, -1));
    }
    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimum) {
        eval(in()) = eval(in());
    }
};

struct shift_acc_backward_fill_and_flush {

    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0, 0, 1 > > in;

    typedef boost::mpl::vector< in > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_low) {
        eval(in()) = eval(in()) + eval(in(0, 0, 1));
    }
    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximum) {
        eval(in()) = eval(in());
    }
};

struct copy_fill {

    typedef accessor< 0, enumtype::inout > in;

    typedef boost::mpl::vector< in > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kfull) {
        eval(in()) = eval(in());
    }
};

TEST_F(kcachef, fill_and_flush_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, 0) = m_inv(i, j, 0);
            for (uint_t k = 1; k < m_d3; ++k) {
                m_refv(i, j, k) = m_inv(i, j, k) + m_refv(i, j, k-1);
            }
        }
    }

    typedef arg< 0, storage_t > p_in;

    typedef boost::mpl::vector< p_in > accessor_list;
    gridtools::aggregator_type< accessor_list > domain((p_in() = m_in));

    auto kcache_stencil = gridtools::make_computation< backend_t >(
        domain,
        m_grid,
        gridtools::make_multistage // mss_descriptor
        (execute< forward >(),
            define_caches(cache< K, cache_io_policy::fill_and_flush, kfull >(p_in())),
            gridtools::make_stage< shift_acc_forward_fill_and_flush >(p_in())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    m_in.sync();
    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_in, halos));
    kcache_stencil->finalize();
}

TEST_F(kcachef, fill_and_flush_backward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1);
            for (int_t k = m_d3 - 2; k >= 0; --k) {
                m_refv(i, j, k) = m_refv(i, j, k + 1) + m_inv(i, j, k);
            }
        }
    }

    typedef arg< 0, storage_t > p_in;

    typedef boost::mpl::vector< p_in > accessor_list;
    gridtools::aggregator_type< accessor_list > domain((p_in() = m_in));

    auto kcache_stencil = gridtools::make_computation< backend_t >(
        domain,
        m_grid,
        gridtools::make_multistage
        (execute< backward >(),
            define_caches(cache< K, cache_io_policy::fill_and_flush, kfull >(p_in())),
            gridtools::make_stage< shift_acc_backward_fill_and_flush >(p_in())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    m_in.sync();
    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_in, halos));
    kcache_stencil->finalize();
}

TEST_F(kcachef, fill_copy_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            for (uint_t k = 0; k < m_d3; ++k) {
                m_refv(i, j, k) = m_inv(i, j, k);
            }
        }
    }

    typedef arg< 0, storage_t > p_in;

    typedef boost::mpl::vector< p_in > accessor_list;
    gridtools::aggregator_type< accessor_list > domain((p_in() = m_in));

    auto kcache_stencil =
        gridtools::make_computation< backend_t >(domain,
            m_grid,
            gridtools::make_multistage
            (execute< forward >(),
                                                     define_caches(cache< K, cache_io_policy::fill_and_flush, kfull >(p_in())),
                                                     gridtools::make_stage< copy_fill >(p_in())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    m_in.sync();
    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_in, halos));
    kcache_stencil->finalize();
}
