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
#include <stencil-composition/stencil-composition.hpp>

#include "../../../examples/Options.hpp"
#include "test_cxx11_domain_reassign.hpp"
#include <tools/verifier.hpp>

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace domain_reassign {

    struct test_functor {

        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };
    typedef interval< level< 0, -2 >, level< 1, 1 > > axis;

    gt_example::gt_example(uint_t d1, uint_t d2, uint_t d3, storage_t in, storage_t out) {
        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        aggregator_type< accessor_list > domain(in, out);

        m_stencil = make_computation< gridtools::BACKEND >(domain,
            grid,
            make_multistage // mss_descriptor
            (execute< forward >(), make_stage< test_functor >(p_in(), p_out())));

        m_stencil->ready();
        m_stencil->steady();
    }
    gt_example::~gt_example() { m_stencil->finalize(); }

    void gt_example::run(storage_t in, storage_t out) {

        m_stencil->reassign(in, out);
        m_stencil->run();
    }

    void gt_example::run_plch(storage_t in, storage_t out) {

        m_stencil->reassign((p_in() = in), (p_out() = out));
        m_stencil->run();
    }

    void gt_example::run_on(storage_t in, storage_t out) { m_stencil->run(in, out); }

    void gt_example::run_on_plch(storage_t in, storage_t out) { m_stencil->run((p_in() = in), (p_out() = out)); }
}

using namespace domain_reassign;

class ReassignDomain : public ::testing::Test {
  protected:
    const gridtools::uint_t m_d1, m_d2, m_d3;

    gridtools::halo_descriptor m_di, m_dj;

    gridtools::grid< axis > m_grid;
    storage_info_t m_meta;
    storage_t m_in1, m_out1, m_in2, m_out2;
    gridtools::data_view< storage_t, gridtools::access_mode::ReadWrite > m_in1v, m_out1v, m_in2v, m_out2v;
    gt_example m_stex;
    array< array< uint_t, 2 >, 3 > m_halos;
    verifier m_verif;

    ReassignDomain()
        : m_d1(6), m_d2(6), m_d3(10), m_di{0, 0, 0, m_d1 - 1, m_d1}, m_dj{0, 0, 0, m_d2 - 1, m_d2}, m_grid(m_di, m_dj),
          m_meta(m_d1, m_d2, m_d3), m_in1(m_meta, [](int i, int j, int k) { return i + j + k + 3; }, "in"),
          m_in2(m_meta, [](int i, int j, int k) { return i + j + k + 7; }, "in2"), m_out1(m_meta, -1., "out"),
          m_out2(m_meta, -1., "out"), m_in1v(make_host_view(m_in2)), m_in2v(make_host_view(m_in2)),
          m_out1v(make_host_view(m_out1)), m_out2v(make_host_view(m_out2)), m_stex(m_d1, m_d2, m_d3, m_in1, m_out1),
          m_halos{{{0, 0}, {0, 0}, {0, 0}}},
#if FLOAT_PRECISION == 4
          m_verif(1e-6)
#else
          m_verif(1e-10)
#endif
    {
        m_grid.value_list[0] = 0;
        m_grid.value_list[1] = m_d3 - 1;
        sync();
    }

    void sync() {
        m_out1.sync();
        m_out2.sync();
        m_in1.sync();
        m_in2.sync();
    }

    storage_t create_new_field(std::string name) { return storage_t(m_meta, -1, name); }
};

TEST_F(ReassignDomain, TestRun) {
    m_stex.run(m_in1, m_out1);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_in1, m_out1, m_halos));

    m_stex.run(m_in2, m_out2);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_in2, m_out2, m_halos));
}

TEST_F(ReassignDomain, TestRunPlchr) {
    m_stex.run_plch(m_in1, m_out1);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_in1, m_out1, m_halos));

    m_stex.run_plch(m_in2, m_out2);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_in2, m_out2, m_halos));
}

TEST_F(ReassignDomain, TestRunOn) {
    m_stex.run_on(m_in1, m_out1);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_in1, m_out1, m_halos));

    m_stex.run_on(m_in2, m_out2);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_in2, m_out2, m_halos));
}

TEST_F(ReassignDomain, TestRunOnPlchr) {
    m_stex.run_on_plch(m_in1, m_out1);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_in1, m_out1, m_halos));

    m_stex.run_on_plch(m_in2, m_out2);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_in2, m_out2, m_halos));
}
