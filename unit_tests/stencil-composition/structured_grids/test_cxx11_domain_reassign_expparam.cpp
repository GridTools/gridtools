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

#include <vector>

#include <boost/mpl/vector.hpp>
#include <gtest/gtest.h>

#include <common/defs.hpp>
#include <common/halo_descriptor.hpp>
#include <stencil-composition/arg.hpp>
#include <stencil-composition/computation.hpp>
#include <stencil-composition/grid.hpp>
#include <stencil-composition/make_computation.hpp>
#include <stencil-composition/make_stage.hpp>
#include <stencil-composition/make_stencils.hpp>
#include <stencil-composition/expandable_parameters/expand_factor.hpp>
#include <stencil-composition/expandable_parameters/vector_accessor.hpp>
#include <tools/verifier.hpp>
#include "backend_select.hpp"

namespace gridtools {
    struct test_functor {
        using in = vector_accessor< 0, enumtype::in, extent<> >;
        using out = vector_accessor< 1, enumtype::inout, extent<> >;
        using arg_list = boost::mpl::vector< in, out >;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    class fixture : public ::testing::Test {
      public:
        using storage_info_t = backend_t::storage_traits_t::storage_info_t< 0, 3 >;
        using storage_t = backend_t::storage_traits_t::data_store_t< gridtools::float_type, storage_info_t >;
        using p_tmp = tmp_arg< 2, std::vector< storage_t > >;

        const uint_t m_d1 = 6, m_d2 = 6, m_d3 = 10;
        const halo_descriptor m_di = {0, 0, 0, m_d1 - 1, m_d1}, m_dj = {0, 0, 0, m_d2 - 1, m_d2};
        const grid< axis< 1 >::axis_interval_t > m_grid = make_grid(m_di, m_dj, m_d3);
        const storage_info_t m_meta{m_d1, m_d2, m_d3};

      public:
        using p_in = arg< 0, std::vector< storage_t > >;
        using p_out = arg< 1, std::vector< storage_t > >;

        computation< void, p_in, p_out > m_computation =
            make_computation< backend_t >(expand_factor< 2 >{},
                m_grid,
                make_multistage(enumtype::execute< enumtype::forward >(),
                                              make_stage< test_functor >(p_in{}, p_tmp{}),
                                              make_stage< test_functor >(p_tmp{}, p_out{})));

        std::vector< storage_t > make_in(int n) const {
            return {3, {m_meta, [=](int i, int j, int k) { return i + j + k + n; }}};
        }

        std::vector< storage_t > make_out() const { return {3, {m_meta, -1.}}; }

        bool verify(storage_t const &lhs, storage_t const &rhs) {
            lhs.sync();
            rhs.sync();
#if FLOAT_PRECISION == 4
            const double precision = 1e-6;
#else
            const double precision = 1e-10;
#endif
            return verifier(precision).verify(m_grid, lhs, rhs, {{{0, 0}, {0, 0}, {0, 0}}});
        }

        std::vector< bool > verify(std::vector< storage_t > const &lhs, std::vector< storage_t > const &rhs) {
            assert(lhs.size() == rhs.size());
            size_t n = lhs.size();
            std::vector< bool > res(n);
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
}
