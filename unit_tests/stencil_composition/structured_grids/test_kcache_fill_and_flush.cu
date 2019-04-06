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
using namespace expressions;

// These are the stencil operators that compose the multistage stencil in this test
struct shift_acc_forward_fill_and_flush {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(in, extent<0, 0, 0, 0, -1, 0>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody_high) {
        eval(in()) = eval(in()) + eval(in(0, 0, -1));
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimum) {
        eval(in()) = eval(in());
    }
};

struct shift_acc_backward_fill_and_flush {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(in, extent<0, 0, 0, 0, 0, 1>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody_low) {
        eval(in()) = eval(in()) + eval(in(0, 0, 1));
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximum) {
        eval(in()) = eval(in());
    }
};

struct copy_fill {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(in));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kfull) {
        eval(in()) = eval(in());
    }
};

struct scale_fill {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(in));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kfull) {
        eval(in()) = 2 * eval(in());
    }
};

TEST_F(kcachef, fill_and_flush_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            m_refv(i, j, 0) = m_inv(i, j, 0);
            for (uint_t k = 1; k < m_d3; ++k) {
                m_refv(i, j, k) = m_inv(i, j, k) + m_refv(i, j, k - 1);
            }
        }
    }

    typedef arg<0, storage_t> p_in;

    auto kcache_stencil = gridtools::make_computation<backend_t>(m_grid,
        p_in{} = m_in,
        gridtools::make_multistage(execute::forward(),
            define_caches(cache<cache_type::k, cache_io_policy::fill_and_flush>(p_in())),
            gridtools::make_stage<shift_acc_forward_fill_and_flush>(p_in())));

    kcache_stencil.run();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    m_in.sync();
    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_in, halos));
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

    typedef arg<0, storage_t> p_in;

    auto kcache_stencil = gridtools::make_computation<backend_t>(m_grid,
        p_in{} = m_in,
        gridtools::make_multistage(execute::backward(),
            define_caches(cache<cache_type::k, cache_io_policy::fill_and_flush>(p_in())),
            gridtools::make_stage<shift_acc_backward_fill_and_flush>(p_in())));

    kcache_stencil.run();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    m_in.sync();
    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_in, halos));
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

    auto kcache_stencil = gridtools::make_computation<backend_t>(m_grid,
        p_in{} = m_in,
        gridtools::make_multistage(execute::forward(),
            define_caches(cache<cache_type::k, cache_io_policy::fill_and_flush>(p_in())),
            gridtools::make_stage<copy_fill>(p_in())));

    kcache_stencil.run();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    m_in.sync();
    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_in, halos));
}

TEST_F(kcachef, fill_scale_forward) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            for (uint_t k = 0; k < m_d3; ++k) {
                m_refv(i, j, k) = 2 * m_inv(i, j, k);
            }
        }
    }

    typedef arg<0, storage_t> p_in;

    auto kcache_stencil = gridtools::make_computation<backend_t>(m_grid,
        p_in{} = m_in,
        gridtools::make_multistage(execute::forward(),
            define_caches(cache<cache_type::k, cache_io_policy::fill_and_flush>(p_in())),
            gridtools::make_stage<scale_fill>(p_in())));

    kcache_stencil.run();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    m_in.sync();
    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_in, halos));
}

struct do_nothing {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(in, extent<0, 0, 0, 0, -1, 1>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kminimum) {}
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kmaximum) {}
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, kbody) {}
};

TEST_F(kcachef, fill_copy_forward_with_extent) {

    for (uint_t i = 0; i < m_d1; ++i) {
        for (uint_t j = 0; j < m_d2; ++j) {
            for (uint_t k = 0; k < m_d3; ++k) {
                m_refv(i, j, k) = m_inv(i, j, k) = k;
            }
        }
    }
    m_in.sync();
    m_ref.sync();

    typedef arg<0, storage_t> p_in;

    auto kcache_stencil = gridtools::make_computation<backend_t>(m_grid,
        p_in{} = m_in,
        gridtools::make_multistage(execute::forward(),
            define_caches(cache<cache_type::k, cache_io_policy::fill_and_flush>(p_in())),
            gridtools::make_stage<do_nothing>(p_in())));

    kcache_stencil.run();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-10);
#endif
    array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};

    m_in.sync();
    ASSERT_TRUE(verif.verify(m_grid, m_ref, m_in, halos));
}
