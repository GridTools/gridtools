/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
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

struct shift_acc_forward_flush {

    typedef accessor<0, intent::in, extent<>> in;
    typedef accessor<1, intent::inout, extent<0, 0, 0, 0, -1, 0>> out;

    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimum) {
        eval(out()) = eval(in());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody_high) {
        eval(out()) = eval(out(0, 0, -1)) + eval(in());
    }
};

struct shift_acc_backward_flush {

    typedef accessor<0, intent::in, extent<>> in;
    typedef accessor<1, intent::inout, extent<0, 0, 0, 0, 0, 1>> out;

    typedef make_param_list<in, out> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximum) {
        eval(out()) = eval(in());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody_low) {
        eval(out()) = eval(out(0, 0, 1)) + eval(in());
    }
};

TEST_F(kcachef, flush_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, 0) = m_inv(i, j, 0);
            for (uint_t k = 1; k < m_d3; ++k) {
                m_refv(i, j, k) = m_refv(i, j, k - 1) + m_inv(i, j, k);
            }
        }
    }

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto kcache_stencil = make_computation<backend_t>(m_grid,
        p_out() = m_out,
        p_in() = m_in,
        make_multistage(execute::forward(),
            define_caches(cache<cache_type::k, cache_io_policy::flush>(p_out())),
            make_stage<shift_acc_forward_flush>(p_in(), p_out())));

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

TEST_F(kcachef, flush_backward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_inv(i, j, m_d3 - 1) = i + j + m_d3 - 1;
            m_refv(i, j, m_d3 - 1) = m_inv(i, j, m_d3 - 1);
            for (int_t k = m_d3 - 2; k >= 0; --k) {
                m_inv(i, j, k) = i + j + k;
                m_refv(i, j, k) = m_refv(i, j, k + 1) + m_inv(i, j, k);
            }
        }
    }

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;

    auto kcache_stencil = make_computation<backend_t>(m_grid,
        p_out() = m_out,
        p_in() = m_in,
        make_multistage(execute::backward(),
            define_caches(cache<cache_type::k, cache_io_policy::flush>(p_out())),
            make_stage<shift_acc_backward_flush>(p_in(), p_out())));

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
