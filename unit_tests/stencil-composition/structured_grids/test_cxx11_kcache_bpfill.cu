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
#include "gtest/gtest.h"
#include "stencil-composition/stencil-composition.hpp"
#include "kcache_fixture.hpp"
#include "tools/verifier.hpp"

using namespace gridtools;
using namespace enumtype;

struct shift_acc_forward_bpfilll {

    typedef accessor< 0, in, extent< 0, 0, 0, 0, -2, 0 > > in;
    typedef accessor< 1, inout, extent<> > out;

    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimum) {
        eval(out()) = eval(in());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimump1) {
        eval(out()) = eval(out(0, 0, -1));
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_highp1) {
        eval(out()) = eval(out(0, 0, -1)) + eval(out(0, 0, -2));
    }
};

struct shift_acc_backward_bpfilll {

    typedef accessor< 0, in, extent< 0, 0, 0, 0, 0, 2 > > in;
    typedef accessor< 1, inout, extent<> > out;

    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximum_b) {
        eval(out()) = eval(in());
    }
    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximumm1_b) {
        eval(out()) = eval(out(0, 0, 1));
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_low_b) {
        eval(out()) = eval(out(0, 0, 1)) + eval(out(0, 0, 2));
    }
};

struct self_update_forward_bpfilll {

    typedef accessor< 0, inout, extent< 0, 0, 0, 0, -2, 1 > > in;
    typedef inout_accessor< 1 > out;

    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimum) {
        eval(out()) = eval(in());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimump1) {

        eval(out()) = eval(in());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_highp1) {
        eval(in()) = eval(in(0, 0, -1)) + eval(in(0, 0, -2));
        eval(out()) = eval(in());
    }
};

struct self_update_backward_bpfilll {

    typedef accessor< 0, inout, extent< 0, 0, 0, 0, -1, 2 > > in;
    typedef inout_accessor< 1 > out;

    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximum_b) {
        eval(out()) = eval(in());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximumm1_b) {

        eval(out()) = eval(in());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_low_b) {
        eval(in()) = eval(in(0, 0, 1)) + eval(in(0, 0, 2));
        eval(out()) = eval(in());
    }
};

TEST_F(kcachef, bpfilll_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, 0) = m_inv(i, j, 0);
            m_refv(i, j, 1) = m_refv(i, j, 0);

            for (uint_t k = 2; k < m_d3; ++k) {
                m_refv(i, j, k) = m_refv(i, j, k - 1) + m_refv(i, j, k - 2);
            }
        }
    }

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;

    typedef boost::mpl::vector< p_in, p_out > accessor_list;
    aggregator_type< accessor_list > domain((p_in() = m_in), (p_out() = m_out));

    auto kcache_stencil =
        make_computation< backend_t >(domain,
            m_grid,
            make_multistage // mss_descriptor
            (execute< forward >(),
                                          define_caches(cache< K, cache_io_policy::bpfill, kfull >(p_in())),
                                          make_stage< shift_acc_forward_bpfilll >(p_in() // esf_descriptor
                                              ,
                                              p_out())));

    kcache_stencil.steady();

    kcache_stencil.run();

    m_out.sync();
    m_out.reactivate_host_write_views();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));

    kcache_stencil.finalize();
}

TEST_F(kcachef, bpfilll_backward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1);
            m_refv(i, j, m_d3 - 2) = m_refv(i, j, m_d3 - 1);

            for (int_t k = m_d3 - 3; k >= 0; --k) {
                m_refv(i, j, k) = m_refv(i, j, k + 1) + m_refv(i, j, k + 2);
            }
        }
    }

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;

    typedef boost::mpl::vector< p_in, p_out > accessor_list;
    aggregator_type< accessor_list > domain((p_in() = m_in), (p_out() = m_out));

    auto kcache_stencil =
        make_computation< backend_t >(domain,
            m_gridb,
            make_multistage // mss_descriptor
            (execute< backward >(),
                                          define_caches(cache< K, cache_io_policy::bpfill, kfull_b >(p_in())),
                                          make_stage< shift_acc_backward_bpfilll >(p_in() // esf_descriptor
                                              ,
                                              p_out())));

    kcache_stencil.steady();

    kcache_stencil.run();
    m_out.sync();
    m_out.reactivate_host_write_views();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));

    kcache_stencil.finalize();
}

TEST_F(kcachef, bpfilll_selfupdate_forward) {

    auto buff = create_new_field("buff");

    auto buffv = make_host_view(buff);

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, 0) = m_inv(i, j, 0);
            m_refv(i, j, 1) = m_inv(i, j, 1);
            buffv(i, j, 0) = m_inv(i, j, 0);
            buffv(i, j, 1) = m_inv(i, j, 1);

            for (uint_t k = 2; k < m_d3; ++k) {
                m_refv(i, j, k) = m_refv(i, j, k - 1) + m_refv(i, j, k - 2);
            }
        }
    }

    typedef arg< 0, storage_t > p_out;
    typedef arg< 1, storage_t > p_buff;

    typedef boost::mpl::vector< p_out, p_buff > accessor_list;
    aggregator_type< accessor_list > domain((p_buff() = buff), (p_out() = m_out));

    auto kcache_stencil =
        make_computation< backend_t >(domain,
            m_grid,
            make_multistage // mss_descriptor
            (execute< forward >(),
                                          define_caches(cache< K, cache_io_policy::bpfill, kfull >(p_buff())),
                                          make_stage< self_update_forward_bpfilll >(p_buff(), p_out())));

    kcache_stencil.steady();

    kcache_stencil.run();

    m_out.sync();
    m_out.reactivate_host_write_views();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));
    kcache_stencil.finalize();
}

TEST_F(kcachef, bpfilll_selfupdate_backward) {

    auto buff = create_new_field("buff");

    auto buffv = make_host_view(buff);

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1);
            m_refv(i, j, m_d3 - 2) = m_inv(i, j, m_d3 - 2);
            buffv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1);
            buffv(i, j, m_d3 - 2) = m_inv(i, j, m_d3 - 2);

            for (int_t k = m_d3 - 3; k >= 0; --k) {
                m_refv(i, j, k) = m_refv(i, j, k + 1) + m_refv(i, j, k + 2);
            }
        }
    }

    typedef arg< 0, storage_t > p_out;
    typedef arg< 1, storage_t > p_buff;

    typedef boost::mpl::vector< p_out, p_buff > accessor_list;
    aggregator_type< accessor_list > domain((p_buff() = buff), (p_out() = m_out));

    auto kcache_stencil =
        make_computation< backend_t >(domain,
            m_gridb,
            make_multistage // mss_descriptor
            (execute< backward >(),
                                          define_caches(cache< K, cache_io_policy::bpfill, kfull_b >(p_buff())),
                                          make_stage< self_update_backward_bpfilll >(p_buff(), p_out())));

    kcache_stencil.steady();

    kcache_stencil.run();

    m_out.sync();
    m_out.reactivate_host_write_views();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));

    kcache_stencil.finalize();
}
