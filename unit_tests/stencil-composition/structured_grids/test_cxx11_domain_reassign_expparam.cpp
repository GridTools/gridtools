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
#include "test_cxx11_domain_reassign_expparam.hpp"
#include <tools/verifier.hpp>

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace domain_reassign {

    struct test_functor {

        typedef vector_accessor< 0, enumtype::in, extent<> > in;
        typedef vector_accessor< 1, enumtype::inout, extent<> > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };
    typedef interval< level< 0, -1 >, level< 1, 1 > > axis_t;

    gt_example::gt_example(
        uint_t d1, uint_t d2, uint_t d3, std::vector< storage_t > &list_in, std::vector< storage_t > &list_out) {
        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        grid< axis_t > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        aggregator_type< accessor_list > domain(list_in, list_out);

        m_stencil = make_computation< gridtools::BACKEND >(expand_factor< 2 >(),
            domain,
            grid,
            make_multistage // mss_descriptor
            (execute< forward >(),
                                                               make_stage< test_functor >(p_in(), p_tmp()),
                                                               make_stage< test_functor >(p_tmp(), p_out())));

        m_stencil->ready();
        m_stencil->steady();
    }
    gt_example::~gt_example() { m_stencil->finalize(); }

    void gt_example::run(std::vector< storage_t > &in, std::vector< storage_t > &out) {

        m_stencil->reassign(in, out);
        m_stencil->run();
    }

    void gt_example::run_plch(std::vector< storage_t > &in, std::vector< storage_t > &out) {

        m_stencil->reassign((p_in() = in), (p_out() = out));
        m_stencil->run();
    }

    void gt_example::run_on(std::vector< storage_t > &in, std::vector< storage_t > &out) { m_stencil->run(in, out); }

    void gt_example::run_on_output(std::vector< storage_t > &out) { m_stencil->run(p_out() = out); }

    void gt_example::run_on_plch(std::vector< storage_t > &in, std::vector< storage_t > &out) {
        m_stencil->run((p_in() = in), (p_out() = out));
    }
}

using namespace domain_reassign;

class ReassignDomain : public ::testing::Test {
  protected:
    const gridtools::uint_t m_d1, m_d2, m_d3;

    gridtools::halo_descriptor m_di, m_dj;

    gridtools::grid< axis_t > m_grid;
    storage_info_t m_meta;
    storage_t m_tr1_in1, m_tr1_in2, m_tr1_in3, m_tr2_in1, m_tr2_in2, m_tr2_in3, m_tr1_out1, m_tr1_out2, m_tr1_out3,
        m_tr2_out1, m_tr2_out2, m_tr2_out3;
    std::vector< storage_t > m_tr1_in_list, m_tr2_in_list, m_tr1_out_list, m_tr2_out_list;
    gt_example m_stex;
    array< array< uint_t, 2 >, 3 > m_halos;
    verifier m_verif;

    ReassignDomain()
        : m_d1(6), m_d2(6), m_d3(10), m_di{0, 0, 0, m_d1 - 1, m_d1}, m_dj{0, 0, 0, m_d2 - 1, m_d2}, m_grid(m_di, m_dj),
          m_meta(m_d1, m_d2, m_d3), m_tr1_in1(m_meta, [](int i, int j, int k) { return i + j + k + 3; }, "tr1_in1"),
          m_tr1_in2(m_meta, [](int i, int j, int k) { return i + j + k + 3; }, "tr1_in2"),
          m_tr1_in3(m_meta, [](int i, int j, int k) { return i + j + k + 7; }, "tr1_in3"),
          m_tr2_in1(m_meta, [](int i, int j, int k) { return i + j + k + 4; }, "tr2_in1"),
          m_tr2_in2(m_meta, [](int i, int j, int k) { return i + j + k + 1; }, "tr2_in2"),
          m_tr2_in3(m_meta, [](int i, int j, int k) { return i + j + k + 2; }, "tr2_in3"),
          m_tr1_out1(m_meta, -1., "tr1_out1"), m_tr1_out2(m_meta, -1., "tr1_out2"), m_tr1_out3(m_meta, -1., "tr1_out3"),
          m_tr2_out1(m_meta, -1., "tr2_out1"), m_tr2_out2(m_meta, -1., "tr2_out2"), m_tr2_out3(m_meta, -1., "tr2_out3"),
          m_tr1_in_list{m_tr1_in1, m_tr1_in2, m_tr1_in3}, m_tr2_in_list{m_tr2_in1, m_tr2_in2, m_tr2_in3},
          m_tr1_out_list{m_tr1_out1, m_tr1_out2, m_tr1_out3}, m_tr2_out_list{m_tr2_out1, m_tr2_out2, m_tr2_out3},
          m_stex(m_d1, m_d2, m_d3, m_tr1_in_list, m_tr1_out_list), m_halos{{{0, 0}, {0, 0}, {0, 0}}},
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

    void sync(std::vector< storage_t > &fields) {
        for (auto it : fields) {
            it.sync();
        }
    }

    void sync() {
        sync(m_tr1_in_list);
        sync(m_tr2_in_list);
        sync(m_tr1_out_list);
        sync(m_tr2_out_list);
    }

    storage_t create_new_field(std::string name) { return storage_t(m_meta, -1, name); }
};

TEST_F(ReassignDomain, TestRun) {
    m_stex.run(m_tr1_in_list, m_tr1_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in1, m_tr1_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in2, m_tr1_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in3, m_tr1_out3, m_halos));

    m_stex.run(m_tr2_in_list, m_tr2_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in1, m_tr2_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in2, m_tr2_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in3, m_tr2_out3, m_halos));
}

TEST_F(ReassignDomain, TestRunPlchr) {
    m_stex.run_plch(m_tr1_in_list, m_tr1_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in1, m_tr1_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in2, m_tr1_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in3, m_tr1_out3, m_halos));

    m_stex.run_plch(m_tr2_in_list, m_tr2_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in1, m_tr2_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in2, m_tr2_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in3, m_tr2_out3, m_halos));
}

TEST_F(ReassignDomain, TestRunOn) {
    m_stex.run_on(m_tr1_in_list, m_tr1_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in1, m_tr1_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in2, m_tr1_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in3, m_tr1_out3, m_halos));

    m_stex.run_on(m_tr2_in_list, m_tr2_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in1, m_tr2_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in2, m_tr2_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in3, m_tr2_out3, m_halos));
}

TEST_F(ReassignDomain, TestRunOnPlchr) {
    m_stex.run_on_plch(m_tr1_in_list, m_tr1_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in1, m_tr1_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in2, m_tr1_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in3, m_tr1_out3, m_halos));

    m_stex.run_on_plch(m_tr2_in_list, m_tr2_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in1, m_tr2_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in2, m_tr2_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr2_in3, m_tr2_out3, m_halos));
}

TEST_F(ReassignDomain, TestRunOnOutput) {
    m_stex.run_on_plch(m_tr1_in_list, m_tr1_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in1, m_tr1_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in2, m_tr1_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in3, m_tr1_out3, m_halos));

    m_stex.run_on_output(m_tr2_out_list);

    sync();
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in1, m_tr2_out1, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in2, m_tr2_out2, m_halos));
    ASSERT_TRUE(m_verif.verify(m_grid, m_tr1_in3, m_tr2_out3, m_halos));
}
