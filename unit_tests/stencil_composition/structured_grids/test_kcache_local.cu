/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "kcache_fixture.hpp"
#include "gtest/gtest.h"
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/verifier.hpp>

using namespace gridtools;

struct shif_acc_forward {

    typedef accessor<0, intent::in, extent<>> in;
    typedef accessor<1, intent::inout, extent<>> out;
    typedef accessor<2, intent::inout, extent<0, 0, 0, 0, -1, 0>> buff;

    typedef make_param_list<in, out, buff> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimum) {
        eval(buff()) = eval(in());
        eval(out()) = eval(buff());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody_high) {

        eval(buff()) = eval(buff(0, 0, -1)) + eval(in());
        eval(out()) = eval(buff());
    }
};

struct biside_large_kcache_forward {

    typedef accessor<0, intent::in, extent<>> in;
    typedef accessor<1, intent::inout, extent<>> out;
    typedef accessor<2, intent::inout, extent<0, 0, 0, 0, -2, 1>> buff;

    typedef make_param_list<in, out, buff> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimum) {
        eval(buff()) = eval(in());
        eval(buff(0, 0, 1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimump1) {
        eval(buff(0, 0, 1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) * (float_type)0.25;
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody_highp1m1) {
        eval(buff(0, 0, 1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) * (float_type)0.25 + eval(buff(0, 0, -2)) * (float_type)0.12;
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximum) {
        eval(out()) = eval(buff()) + eval(buff(0, 0, -1)) * (float_type)0.25 + eval(buff(0, 0, -2)) * (float_type)0.12;
    }
};

struct biside_large_kcache_backward {

    typedef accessor<0, intent::in, extent<>> in;
    typedef accessor<1, intent::inout, extent<>> out;
    typedef accessor<2, intent::inout, extent<0, 0, 0, 0, -1, 2>> buff;

    typedef make_param_list<in, out, buff> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximum) {
        eval(buff()) = eval(in());
        eval(buff(0, 0, -1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximumm1) {
        eval(buff(0, 0, -1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) * (float_type)0.25;
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody_lowp1) {
        eval(buff(0, 0, -1)) = eval(in()) * (float_type)0.5;
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) * (float_type)0.25 + eval(buff(0, 0, 2)) * (float_type)0.12;
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimum) {
        eval(out()) = eval(buff()) + eval(buff(0, 0, 1)) * (float_type)0.25 + eval(buff(0, 0, 2)) * (float_type)0.12;
    }
};

struct shif_acc_backward {

    typedef accessor<0, intent::in, extent<>> in;
    typedef accessor<1, intent::inout, extent<>> out;
    typedef accessor<2, intent::inout, extent<0, 0, 0, 0, 0, 1>> buff;

    typedef make_param_list<in, out, buff> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximum) {
        eval(buff()) = eval(in());
        eval(out()) = eval(buff());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody_low) {
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

    arg<0> p_in;
    arg<1> p_out;
    tmp_arg<2, float_type> p_buff;

    compute<backend_t>(m_grid,
        p_in = m_in,
        p_out = m_out,
        make_multistage(execute::forward(),
            define_caches(cache<cache_type::k>(p_buff)),
            make_stage<shif_acc_forward>(p_in, p_out, p_buff)));

    m_out.sync();
    m_out.reactivate_host_write_views();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));
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

    arg<0> p_in;
    arg<1> p_out;
    tmp_arg<2, float_type> p_buff;

    compute<backend_t>(m_grid,
        p_in = m_in,
        p_out = m_out,
        make_multistage(execute::backward(),
            define_caches(cache<cache_type::k>(p_buff)),
            make_stage<shif_acc_backward>(p_in, p_out, p_buff)));

    m_out.sync();
    m_out.reactivate_host_write_views();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));
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

    arg<0> p_in;
    arg<1> p_out;
    tmp_arg<2, float_type> p_buff;

    compute<backend_t>(m_grid,
        p_in = m_in,
        p_out = m_out,
        make_multistage(execute::forward(),
            define_caches(cache<cache_type::k>(p_buff)),
            make_stage<biside_large_kcache_forward>(p_in, p_out, p_buff)));

    m_out.sync();
    m_out.reactivate_host_write_views();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));
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

    arg<0> p_in;
    arg<1> p_out;
    tmp_arg<2, float_type> p_buff;

    compute<backend_t>(m_grid,
        p_in = m_in,
        p_out = m_out,
        make_multistage(execute::backward(),
            define_caches(cache<cache_type::k>(p_buff)),
            make_stage<biside_large_kcache_backward>(p_in, p_out, p_buff)));

    m_out.sync();
    m_out.reactivate_host_write_views();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_out, halos));
}
