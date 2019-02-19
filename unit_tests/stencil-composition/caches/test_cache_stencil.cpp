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
#include <boost/mpl/equal.hpp>
#include <boost/shared_ptr.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/make_computation.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/verifier.hpp>

constexpr int halo_size = 1;

namespace test_cache_stencil {

    using namespace gridtools;
    using namespace execute;

    struct functor1 {
        typedef accessor<0, intent::in> in;
        typedef accessor<1, intent::inout> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    struct functor2 {
        typedef accessor<0, intent::in, extent<-1, 1, -1, 1>> in;
        typedef accessor<1, intent::inout> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) =
                (eval(in(-1, 0, 0)) + eval(in(1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0))) / (float_type)4.0;
        }
    };

    struct functor3 {
        typedef accessor<0, intent::in> in;
        typedef accessor<1, intent::inout> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(in()) + 1;
        }
    };

    typedef backend_t::storage_traits_t::storage_info_t<0, 3, halo<halo_size, halo_size, 0>> storage_info_t;
    typedef backend_t::storage_traits_t::data_store_t<float_type, storage_info_t> storage_t;

    typedef arg<0, storage_t> p_in;
    typedef arg<1, storage_t> p_out;
    typedef tmp_arg<2, storage_t> p_buff;
    typedef tmp_arg<3, storage_t> p_buff_2;
    typedef tmp_arg<4, storage_t> p_buff_3;
} // namespace test_cache_stencil

using namespace gridtools;
using namespace execute;
using namespace test_cache_stencil;

class cache_stencil : public ::testing::Test {
  protected:
    const uint_t m_d1, m_d2, m_d3;

    halo_descriptor m_di, m_dj;

    gridtools::grid<axis<1>::axis_interval_t> m_grid;
    storage_info_t m_meta;
    storage_t m_in, m_out;

    cache_stencil()
        : m_d1(128), m_d2(128), m_d3(30), m_di{halo_size, halo_size, halo_size, m_d1 - halo_size - 1, m_d1},
          m_dj{halo_size, halo_size, halo_size, m_d2 - halo_size - 1, m_d2}, m_grid(make_grid(m_di, m_dj, m_d3)),
          m_meta(m_d1 + 2 * halo_size, m_d2 + 2 * halo_size, m_d3), m_in(m_meta, 0.), m_out(m_meta, 0.) {}

    virtual void SetUp() {
        m_in = storage_t(m_meta, 0.);
        m_out = storage_t(m_meta, 0.);
        auto m_inv = make_host_view(m_in);
        for (int i = m_di.begin(); i < m_di.end(); ++i) {
            for (int j = m_dj.begin(); j < m_dj.end(); ++j) {
                for (int k = 0; k < m_d3; ++k) {
                    m_inv(i, j, k) = i + j * 100 + k * 10000;
                }
            }
        }
    }
};

TEST_F(cache_stencil, ij_cache) {
    SetUp();

    auto stencil = make_computation<backend_t>(m_grid,
        p_in() = m_in,
        p_out() = m_out,
        make_multistage // mss_descriptor
        (execute::parallel(),
            define_caches(cache<cache_type::ij, cache_io_policy::local>(p_buff())),
            make_stage<functor1>(p_in(), p_buff()),
            make_stage<functor1>(p_buff(), p_out())));

    stencil.run();

    stencil.sync_bound_data_stores();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array<array<uint_t, 2>, 3> halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
    ASSERT_TRUE(verif.verify(m_grid, m_in, m_out, halos));
}

TEST_F(cache_stencil, ij_cache_offset) {
    SetUp();
    storage_info_t meta_(m_d1 + 2 * halo_size, m_d2 + 2 * halo_size, m_d3);
    storage_t ref(meta_, 0.0);
    auto m_inv = make_host_view(m_in);
    auto refv = make_host_view(ref);
    for (int i = halo_size; i < m_d1 - halo_size; ++i) {
        for (int j = halo_size; j < m_d2 - halo_size; ++j) {
            for (int k = 0; k < m_d3; ++k) {
                refv(i, j, k) = (m_inv(i - 1, j, k) + m_inv(i + 1, j, k) + m_inv(i, j - 1, k) + m_inv(i, j + 1, k)) /
                                (float_type)4.0;
            }
        }
    }

    auto stencil = make_computation<backend_t>(m_grid,
        p_in() = m_in,
        p_out() = m_out,
        make_multistage // mss_descriptor
        (execute::parallel(),
            // define_caches(cache< IJ, cache_io_policy::local >(p_buff())),
            make_stage<functor1>(p_in(), p_buff()), // esf_descriptor
            make_stage<functor2>(p_buff(), p_out()) // esf_descriptor
            ));

    stencil.run();

    stencil.sync_bound_data_stores();

#if GT_FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array<array<uint_t, 2>, 3> halos{{{halo_size, halo_size}, {halo_size, halo_size}, {0, 0}}};
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out, halos));
}

TEST_F(cache_stencil, multi_cache) {
    SetUp();
    storage_info_t meta_(m_d1 + 2 * halo_size, m_d2 + 2 * halo_size, m_d3);
    storage_t ref(meta_, 0.0);
    auto m_inv = make_host_view(m_in);
    auto refv = make_host_view(ref);

    for (int i = halo_size; i < m_d1 - halo_size; ++i) {
        for (int j = halo_size; j < m_d2 - halo_size; ++j) {
            for (int k = 0; k < m_d3; ++k) {
                refv(i, j, k) = (m_inv(i, j, k) + 4);
            }
        }
    }

    auto stencil = make_computation<backend_t>(m_grid,
        p_in() = m_in,
        p_out() = m_out,
        make_multistage // mss_descriptor
        (execute::parallel(),
            // test if define_caches works properly with multiple vectors of caches.
            // in this toy example two vectors are passed (IJ cache vector for p_buff
            // and p_buff_2, IJ cache vector for p_buff_3)
            define_caches(cache<cache_type::ij, cache_io_policy::local>(p_buff(), p_buff_2()),
                cache<cache_type::ij, cache_io_policy::local>(p_buff_3())),
            make_stage<functor3>(p_in(), p_buff()),       // esf_descriptor
            make_stage<functor3>(p_buff(), p_buff_2()),   // esf_descriptor
            make_stage<functor3>(p_buff_2(), p_buff_3()), // esf_descriptor
            make_stage<functor3>(p_buff_3(), p_out())     // esf_descriptor
            ));
    stencil.run();

    stencil.sync_bound_data_stores();

    verifier verif(1e-13);
    array<array<uint_t, 2>, 3> halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out, halos));
}
