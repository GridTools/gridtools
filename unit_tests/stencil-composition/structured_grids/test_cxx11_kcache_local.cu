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

struct shif_acc_forward {

    typedef accessor< 0, ::in, extent<> > in;
    typedef accessor< 1, ::inout, extent<> > out;
    typedef accessor< 2, ::inout, extent< 0, 0, 0, 0, -1, 0 > > buff;

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

struct biside_large_kcache_forward {

    typedef accessor< 0, ::in, extent<> > in;
    typedef accessor< 1, ::inout, extent<> > out;
    typedef accessor< 2, ::inout, extent< 0, 0, 0, 0, -2, 1 > > buff;

    typedef boost::mpl::vector< in, out, buff > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimum) {
        eval(buff()) = eval(in());
        eval(buff(0, 0, 1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimump1) {
        eval(buff(0, 0, 1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) * (float_type)0.25;
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_highp1m1) {
        eval(buff(0, 0, 1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) * (float_type)0.25 + eval(buff(0, 0, -2)) * (float_type)0.12;
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximum) {
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) * (float_type)0.25 + eval(buff(0, 0, -2)) * (float_type)0.12;
    }
};

struct biside_large_kcache_backward {

    typedef accessor< 0, ::in, extent<> > in;
    typedef accessor< 1, ::inout, extent<> > out;
    typedef accessor< 2, ::inout, extent< 0, 0, 0, 0, -1, 2 > > buff;

    typedef boost::mpl::vector< in, out, buff > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximum_b) {
        eval(buff()) = eval(in());
        eval(buff(0, 0, -1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff());
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kmaximumm1_b) {
        eval(buff(0, 0, -1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) * (float_type)0.25;
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kbody_lowp1_b) {
        eval(buff(0, 0, -1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) * (float_type)0.25 + eval(buff(0, 0, 2)) * (float_type)0.12;
    }

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, kminimum_b) {
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) * (float_type)0.25 + eval(buff(0, 0, 2)) * (float_type)0.12;
    }
};

struct shif_acc_backward {

    typedef accessor< 0, ::in, extent<> > in;
    typedef accessor< 1, ::inout, extent<> > out;
    typedef accessor< 2, ::inout, extent< 0, 0, 0, 0, 0, 1 > > buff;

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

TEST_F(kcachef, local_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, 0) = m_inv(i, j, 0);
            for (uint_t k = 1; k < m_d3; ++k) {
                m_refv(i, j, k) = m_refv(i, j, k - 1) + m_inv(i, j, k);
                m_outv(i, j, k) = -1;
            }
        }
    }

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef tmp_arg< 2, storage_t > p_buff;

    typedef boost::mpl::vector< p_in, p_out, p_buff > accessor_list;
    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
    // that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in
    // order. (I don't particularly like this)
    gridtools::aggregator_type< accessor_list > domain((p_in() = m_in), (p_out() = m_out));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::grid<axis> grid(2,d1-2,2,d2-2);

    auto kcache_stencil =
        gridtools::make_computation< backend_t >(domain,
            m_grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                                                     define_caches(cache< K, cache_io_policy::local, kfull >(p_buff())),
                                                     gridtools::make_stage< shif_acc_forward >(p_in() // esf_descriptor
                                                         ,
                                                         p_out(),
                                                         p_buff())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

    m_out.sync();
    m_out.reactivate_host_write_views();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));

    kcache_stencil->finalize();
}

TEST_F(kcachef, local_backward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1);
            for (int_t k = m_d3 - 2; k >= 0; --k) {
                m_refv(i, j, k) = m_refv(i, j, k + 1) + m_inv(i, j, k);
            }
        }
    }

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef tmp_arg< 2, storage_t > p_buff;

    typedef boost::mpl::vector< p_in, p_out, p_buff > accessor_list;
    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
    // that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in
    // order. (I don't particularly like this)
    gridtools::aggregator_type< accessor_list > domain((p_in() = m_in), (p_out() = m_out));

    auto kcache_stencil =
        gridtools::make_computation< backend_t >(domain,
            m_grid,
            gridtools::make_multistage // mss_descriptor
            (execute< backward >(),
                                                     define_caches(cache< K, cache_io_policy::local, kfull >(p_buff())),
                                                     gridtools::make_stage< shif_acc_backward >(p_in() // esf_descriptor
                                                         ,
                                                         p_out(),
                                                         p_buff())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

    m_out.sync();
    m_out.reactivate_host_write_views();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));

    kcache_stencil->finalize();
}

TEST_F(kcachef, biside_forward) {

    auto buff = create_new_field("buff");
    auto buffv = make_host_view(buff);

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            buffv(i, j, 0) = m_inv(i, j, 0);
            buffv(i, j, 1) = m_inv(i, j, 0) * (float_type)0.5;
            m_refv(i, j, 0) = m_inv(i, j, 0);

            buffv(i, j, 2) = m_inv(i, j, 1) * (float_type)0.5;
            m_refv(i, j, 1) = buffv(i, j, 1) + (float_type)0.25 * buffv(i, j, 0);
            for (uint_t k = 2; k < m_d3; ++k) {
                if (k != m_d3 - 1)
                    buffv(i, j, k + 1) = m_inv(i, j, k) * (float_type)0.5;
                m_refv(i, j, k) =
                    buffv(i, j, k) + (float_type)0.25 * buffv(i, j, k - 1) + (float_type)0.12 * buffv(i, j, k - 2);
            }
        }
    }

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef tmp_arg< 2, storage_t > p_buff;

    typedef boost::mpl::vector< p_in, p_out, p_buff > accessor_list;
    gridtools::aggregator_type< accessor_list > domain((p_in() = m_in), (p_out() = m_out));

    auto kcache_stencil = gridtools::make_computation< backend_t >(
        domain,
        m_grid,
        gridtools::make_multistage // mss_descriptor
        (execute< forward >(),
            define_caches(cache< K, cache_io_policy::local, kfull >(p_buff())),
            gridtools::make_stage< biside_large_kcache_forward >(p_in() // esf_descriptor
                ,
                p_out(),
                p_buff())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

    m_out.sync();
    m_out.reactivate_host_write_views();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));

    kcache_stencil->finalize();
}

TEST_F(kcachef, biside_backward) {

    auto buff = create_new_field("buff");
    auto buffv = make_host_view(buff);

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            buffv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1);
            buffv(i, j, m_d3 - 2) = m_inv(i, j, m_d3 - 1) * (float_type)0.5;
            m_refv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1);

            buffv(i, j, m_d3 - 3) = m_inv(i, j, m_d3 - 2) * (float_type)0.5;
            m_refv(i, j, m_d3 - 2) = buffv(i, j, m_d3 - 2) + (float_type)0.25 * buffv(i, j, m_d3 - 1);

            for (int_t k = m_d3 - 3; k >= 0; --k) {
                if (k != 0)
                    buffv(i, j, k - 1) = m_inv(i, j, k) * (float_type)0.5;
                m_refv(i, j, k) =
                    buffv(i, j, k) + (float_type)0.25 * buffv(i, j, k + 1) + (float_type)0.12 * buffv(i, j, k + 2);
            }
        }
    }

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef tmp_arg< 2, storage_t > p_buff;

    typedef boost::mpl::vector< p_in, p_out, p_buff > accessor_list;
    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
    // that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in
    // order. (I don't particularly like this)
    gridtools::aggregator_type< accessor_list > domain((p_in() = m_in), (p_out() = m_out));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::grid<axis> grid(2,d1-2,2,d2-2);

    auto kcache_stencil = gridtools::make_computation< backend_t >(
        domain,
        m_gridb,
        gridtools::make_multistage // mss_descriptor
        (execute< backward >(),
            define_caches(cache< K, cache_io_policy::local, kfull >(p_buff())),
            gridtools::make_stage< biside_large_kcache_backward >(p_in() // esf_descriptor
                ,
                p_out(),
                p_buff())));

    kcache_stencil->ready();

    kcache_stencil->steady();

    kcache_stencil->run();

    m_out.sync();
    m_out.reactivate_host_write_views();

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));

    kcache_stencil->finalize();
}
