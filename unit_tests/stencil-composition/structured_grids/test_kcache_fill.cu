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
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/verifier.hpp>

using namespace gridtools;

// These are the stencil operators that compose the multistage stencil in this test
struct shift_acc_forward_fill {

    typedef accessor<0, intent::in, extent<0, 0, 0, 0, -1, 1>> in;
    typedef accessor<1, intent::inout, extent<>> out;

    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimum) {
        eval(out()) = eval(in()) + eval(in(0, 0, 1));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody) {
        eval(out()) = eval(in(0, 0, -1)) + eval(in()) + eval(in(0, 0, 1));
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximum) {
        eval(out()) = eval(in(0, 0, -1)) + eval(in());
    }
};

struct shift_acc_backward_fill {

    typedef accessor<0, intent::in, extent<0, 0, 0, 0, -1, 1>> in;
    typedef accessor<1, intent::inout, extent<>> out;

    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximum) {
        eval(out()) = eval(in()) + eval(in(0, 0, -1));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody) {
        eval(out()) = eval(in(0, 0, 1)) + eval(in()) + eval(in(0, 0, -1));
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimum) {
        eval(out()) = eval(in()) + eval(in(0, 0, 1));
    }
};

struct copy_fill {

    typedef accessor<0, intent::in> in;
    typedef accessor<1, intent::inout, extent<>> out;

    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kfull) {
        eval(out()) = eval(in());
    }
};

TEST_F(kcachef, fill_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, 0) = m_inv(i, j, 0) + m_inv(i, j, 1);
            for (uint_t k = 1; k < m_d3 - 1; ++k) {
                m_refv(i, j, k) = m_inv(i, j, k - 1) + m_inv(i, j, k) + m_inv(i, j, k + 1);
            }
            m_refv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1) + m_inv(i, j, m_d3 - 2);
        }
    }

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto kcache_stencil = gridtools::make_computation<backend_t>(m_grid,
        p_out() = m_out,
        p_in() = m_in,
        gridtools::make_multistage(execute::forward(),
            define_caches(cache<cache_type::k, cache_io_policy::fill>(p_in())),
            gridtools::make_stage<shift_acc_forward_fill>(p_in(), p_out())));

    kcache_stencil.run();

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

TEST_F(kcachef, fill_backward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1) + m_inv(i, j, m_d3 - 2);
            for (int_t k = m_d3 - 2; k >= 1; --k) {
                m_refv(i, j, k) = m_inv(i, j, k + 1) + m_inv(i, j, k) + m_inv(i, j, k - 1);
            }
            m_refv(i, j, 0) = m_inv(i, j, 1) + m_inv(i, j, 0);
        }
    }

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto kcache_stencil = gridtools::make_computation<backend_t>(m_grid,
        p_out() = m_out,
        p_in() = m_in,
        gridtools::make_multistage(execute::backward(),
            define_caches(cache<cache_type::k, cache_io_policy::fill>(p_in())),
            gridtools::make_stage<shift_acc_backward_fill>(p_in(), p_out())));

    kcache_stencil.run();

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

TEST_F(kcachef, fill_copy_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            for (uint_t k = 0; k < m_d3; ++k) {
                m_refv(i, j, k) = m_inv(i, j, k);
            }
        }
    }

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto kcache_stencil = gridtools::make_computation<backend_t>(m_grid,
        p_out() = m_out,
        p_in() = m_in,
        gridtools::make_multistage(execute::forward(),
            define_caches(cache<cache_type::k, cache_io_policy::fill>(p_in())),
            gridtools::make_stage<copy_fill>(p_in(), p_out())));

    kcache_stencil.run();

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
