/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <vector>

#include <boost/mpl/vector.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/halo_descriptor.hpp>
#include <gridtools/stencil-composition/accessor.hpp>
#include <gridtools/stencil-composition/arg.hpp>
#include <gridtools/stencil-composition/computation.hpp>
#include <gridtools/stencil-composition/expandable_parameters/expand_factor.hpp>
#include <gridtools/stencil-composition/grid.hpp>
#include <gridtools/stencil-composition/make_computation.hpp>
#include <gridtools/stencil-composition/make_stage.hpp>
#include <gridtools/stencil-composition/make_stencils.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/verifier.hpp>

namespace gridtools {
    struct test_functor {
        using in = accessor<0, intent::in, extent<>>;
        using out = accessor<1, intent::inout, extent<>>;
        using param_list = make_param_list<in, out>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    class fixture : public ::testing::Test {
      public:
        using storage_info_t = backend_t::storage_traits_t::storage_info_t<0, 3>;
        using storage_t = backend_t::storage_traits_t::data_store_t<float_type, storage_info_t>;
        using p_tmp = tmp_arg<2, std::vector<storage_t>>;

        const uint_t m_d1 = 6, m_d2 = 6, m_d3 = 10;
        const halo_descriptor m_di = {0, 0, 0, m_d1 - 1, m_d1}, m_dj = {0, 0, 0, m_d2 - 1, m_d2};
        const grid<axis<1>::axis_interval_t> m_grid = make_grid(m_di, m_dj, m_d3);
        const storage_info_t m_meta{m_d1, m_d2, m_d3};

      public:
        using p_in = arg<0, std::vector<storage_t>>;
        using p_out = arg<1, std::vector<storage_t>>;

        computation<p_in, p_out> m_computation;

        fixture()
            : m_computation{make_computation<backend_t>(expand_factor<2>{},
                  m_grid,
                  make_multistage(execute::forward(),
                      make_stage<test_functor>(p_in{}, p_tmp{}),
                      make_stage<test_functor>(p_tmp{}, p_out{})))} {}

        std::vector<storage_t> make_in(int n) const {
            return {3, {m_meta, [=](int i, int j, int k) { return i + j + k + n; }}};
        }

        std::vector<storage_t> make_out() const { return {3, {m_meta, -1.}}; }

        bool verify(storage_t const &lhs, storage_t const &rhs) {
            lhs.sync();
            rhs.sync();
#if GT_FLOAT_PRECISION == 4
            const double precision = 1e-6;
#else
            const double precision = 1e-10;
#endif
            return verifier(precision).verify(m_grid, lhs, rhs, {{{0, 0}, {0, 0}, {0, 0}}});
        }

        std::vector<bool> verify(std::vector<storage_t> const &lhs, std::vector<storage_t> const &rhs) {
            assert(lhs.size() == rhs.size());
            size_t n = lhs.size();
            std::vector<bool> res(n);
            for (size_t i = 0; i != n; ++i)
                res[i] = verify(lhs[i], rhs[i]);
            return res;
        }
    };

    TEST_F(fixture, run) {
        auto in = make_in(3);
        auto out = make_out();

        m_computation.run(p_in{} = in, p_out{} = out);
        for (bool res : verify(in, out))
            EXPECT_TRUE(res);
        in = make_in(7);
        m_computation.run(p_in{} = in, p_out{} = out);
        for (bool res : verify(in, out))
            EXPECT_TRUE(res);
    }
} // namespace gridtools
