/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "gtest/gtest.h"

#include "common/defs.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "stencil-composition/make_computation.hpp"
#include "tools/verifier.hpp"

constexpr int halo_size = 1;

namespace test_cache_stencil {

    using namespace gridtools;
    using namespace enumtype;

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > x_interval;
    typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, 1 > > axis;

    struct functor1 {
        typedef accessor< 0, enumtype::in > in;
        typedef accessor< 1, enumtype::inout > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    struct functor2 {
        typedef accessor< 0, enumtype::in, extent< -1, 1, -1, 1 > > in;
        typedef accessor< 1, enumtype::inout > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) =
                (eval(in(-1, 0, 0)) + eval(in(1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0))) / (float_type)4.0;
        }
    };

    struct functor3 {
        typedef accessor< 0, enumtype::in > in;
        typedef accessor< 1, enumtype::inout > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = eval(in()) + 1;
        }
    };

#ifdef __CUDACC__
#define BACKEND backend< Cuda, structured, Block >
#else
#define BACKEND backend< Host, structured, Block >
#endif


    typedef BACKEND::storage_traits_t::storage_info_t<0, 3, halo<halo_size,halo_size,0> > storage_info_t;
    typedef BACKEND::storage_traits_t::data_store_t<float_type, storage_info_t> storage_t;

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef arg< 2, storage_t, true > p_buff;
    typedef arg< 3, storage_t, true > p_buff_2;
    typedef arg< 4, storage_t, true > p_buff_3;
}

using namespace gridtools;
using namespace enumtype;
using namespace test_cache_stencil;

class cache_stencil : public ::testing::Test {
  protected:
    const uint_t m_d1, m_d2, m_d3;

    array< uint_t, 5 > m_di, m_dj;

    gridtools::grid< axis > m_grid;
    storage_info_t m_meta;
    storage_t m_in, m_out;

    cache_stencil()
        : m_d1(32), m_d2(32), m_d3(6),
#ifdef CXX11_ENABLED
          m_di{halo_size, halo_size, halo_size, m_d1 - halo_size - 1, m_d1},
          m_dj{halo_size, halo_size, halo_size, m_d2 - halo_size - 1, m_d2},
#else
          m_di(halo_size, halo_size, halo_size, m_d1 - halo_size - 1, m_d1),
          m_dj(halo_size, halo_size, halo_size, m_d2 - halo_size - 1, m_d2),
#endif
          m_grid(m_di, m_dj), m_meta(m_d1, m_d2, m_d3), m_in(m_meta, 0.), m_out(m_meta, 0.) {
        m_grid.value_list[0] = 0;
        m_grid.value_list[1] = m_d3 - 1;
    }

    virtual void SetUp() {
        m_in = storage_t(m_meta, 0.); 
        m_out = storage_t(m_meta, 0.);
        auto m_inv = make_host_view(m_in);
        for (int i = m_di[2]; i < m_di[3]; ++i) {
            for (int j = m_dj[2]; j < m_dj[3]; ++j) {
                for (int k = 0; k < m_d3; ++k) {
                    m_inv(i, j, k) = i + j * 100 + k * 10000;
                }
            }
        }
    }
};

TEST_F(cache_stencil, ij_cache) {
    SetUp();
    typedef boost::mpl::vector3< p_in, p_out, p_buff > accessor_list;
    gridtools::aggregator_type< accessor_list > domain(m_in, m_out);

    auto pstencil = make_computation< gridtools::BACKEND >(domain,
            m_grid,
            make_multistage // mss_descriptor
            (execute< forward >(),
              define_caches(cache< IJ, local >(p_buff())),
              make_stage< functor1 >(p_in(), p_buff()),
              make_stage< functor1 >(p_buff(), p_out())));

    pstencil->ready();

    pstencil->steady();

    pstencil->run();

    pstencil->finalize();

#ifdef CXX11_ENABLED
#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array< array< uint_t, 2 >, 3 > halos{
        {{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
    ASSERT_TRUE(verif.verify(m_grid, m_in, m_out, halos));
#else
#if FLOAT_PRECISION == 4
    verifier verif(1e-6, halo_size);
#else
    verifier verif(1e-12, halo_size);
#endif
    ASSERT_TRUE(verif.verify(m_grid, m_in, m_out));
#endif
}

TEST_F(cache_stencil, ij_cache_offset) {
    SetUp();
    storage_info_t meta_(m_d1, m_d2, m_d3);
    storage_t ref(meta_, 0.0);
    auto m_inv = make_host_view(m_in);
    auto refv = make_host_view(ref);
    for (int i = halo_size; i < m_d1 - halo_size; ++i) {
        for (int j = halo_size; j < m_d2 - halo_size; ++j) {
            for (int k = 0; k < m_d3; ++k) {
                refv(i, j, k) =
                    (m_inv(i - 1, j, k) + m_inv(i + 1, j, k) + m_inv(i, j - 1, k) + m_inv(i, j + 1, k)) / (float_type)4.0;
            }
        }
    }

    typedef boost::mpl::vector3< p_in, p_out, p_buff > accessor_list;
    gridtools::aggregator_type< accessor_list > domain(m_in, m_out);

    auto pstencil =
            make_computation< gridtools::BACKEND >(domain,
                m_grid,
                make_multistage // mss_descriptor
                (execute< forward >(),
                 define_caches(cache< IJ, local >(p_buff())),
                 make_stage< functor1 >(p_in(), p_buff()), // esf_descriptor
                 make_stage< functor2 >(p_buff(), p_out()) // esf_descriptor
    ));

    pstencil->ready();

    pstencil->steady();

    pstencil->run();

    pstencil->finalize();

#ifdef CXX11_ENABLED
#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array< array< uint_t, 2 >, 3 > halos{
        {{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out, halos));
#else
#if FLOAT_PRECISION == 4
    verifier verif(1e-6, halo_size);
#else
    verifier verif(1e-12, halo_size);
#endif
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out));
#endif
}

TEST_F(cache_stencil, multi_cache) {
    SetUp();
    storage_info_t meta_(m_d1, m_d2, m_d3);
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

    typedef boost::mpl::vector5< p_in, p_out, p_buff, p_buff_2, p_buff_3 > accessor_list;
    gridtools::aggregator_type< accessor_list > domain(m_in, m_out);

    auto stencil = make_computation< gridtools::BACKEND >(
            domain,
            m_grid,
            make_multistage // mss_descriptor
            (execute< forward >(),
                // test if define_caches works properly with multiple vectors of caches.
                // in this toy example two vectors are passed (IJ cache vector for p_buff
                // and p_buff_2, IJ cache vector for p_buff_3)
                define_caches(cache< IJ, local >(p_buff(), p_buff_2()), cache< IJ, local >(p_buff_3())),
                make_stage< functor3 >(p_in(), p_buff()),       // esf_descriptor
                make_stage< functor3 >(p_buff(), p_buff_2()),   // esf_descriptor
                make_stage< functor3 >(p_buff_2(), p_buff_3()), // esf_descriptor
                make_stage< functor3 >(p_buff_3(), p_out())     // esf_descriptor
                ));
    stencil->ready();

    stencil->steady();

    stencil->run();

    stencil->finalize();

#ifdef CXX11_ENABLED
    verifier verif(1e-13);
    array< array< uint_t, 2 >, 3 > halos{
        {{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out, halos));
#else
    verifier verif(1e-13, halo_size);
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out));
#endif
}
