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
#pragma once

#include "stencil-composition/stencil-composition.hpp"

typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, 1 > > axis;

typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > kfull;

typedef gridtools::interval< gridtools::level< 0, 1 >, gridtools::level< 1, -1 > > kbody_high;
typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 0, -1 > > kminimum;
typedef gridtools::interval< gridtools::level< 0, 1 >, gridtools::level< 0, 1 > > kminimump1;
typedef gridtools::interval< gridtools::level< 0, 2 >, gridtools::level< 1, -1 > > kbody_highp1;
typedef gridtools::interval< gridtools::level< 0, 2 >, gridtools::level< 1, -2 > > kbody_highp1m1;
typedef gridtools::interval< gridtools::level< 1, -1 >, gridtools::level< 1, -1 > > kmaximum;
typedef gridtools::interval< gridtools::level< 1, -2 >, gridtools::level< 1, -2 > > kmaximumm1;
typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -2 > > kbody_low;
typedef gridtools::interval< gridtools::level< 0, 1 >, gridtools::level< 1, -2 > > kbody;

typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 2, 1 > > axis_b;
typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 0, -1 > > kminimum_b;
typedef gridtools::interval< gridtools::level< 0, 1 >, gridtools::level< 0, 1 > > kminimump1_b;
typedef gridtools::interval< gridtools::level< 2, -1 >, gridtools::level< 2, -1 > > kmaximum_b;
typedef gridtools::interval< gridtools::level< 1, 1 >, gridtools::level< 2, -2 > > kmaximumm1_b;
typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > kbody_low_b;
typedef gridtools::interval< gridtools::level< 0, 1 >, gridtools::level< 1, -1 > > kbody_lowp1_b;
typedef gridtools::interval< gridtools::level< 1, 2 >, gridtools::level< 1, -1 > > kbody_b;
typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 2, -1 > > kfull_b;

#ifdef __CUDACC__
#define BACKEND_ARCH gridtools::enumtype::Cuda
#define BACKEND backend< BACKEND_ARCH, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Block >
#else
#define BACKEND_ARCH gridtools::enumtype::Host
#ifdef BACKEND_BLOCK
#define BACKEND backend< BACKEND_ARCH, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Block >
#else
#define BACKEND backend< BACKEND_ARCH, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Naive >
#endif
#endif

class kcachef : public ::testing::Test {
  protected:
    typedef gridtools::storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 > storage_info_t;
    typedef gridtools::storage_traits< BACKEND_ARCH >::data_store_t< gridtools::float_type, storage_info_t > storage_t;

    const gridtools::uint_t m_d1, m_d2, m_d3;

    gridtools::halo_descriptor m_di, m_dj;

    gridtools::grid< axis > m_grid;
    gridtools::grid< axis_b > m_gridb;
    storage_info_t m_meta;
    storage_t m_in, m_out, m_ref;
    gridtools::data_view< storage_t, gridtools::access_mode::ReadWrite > m_inv, m_outv, m_refv;

    kcachef()
        : m_d1(6), m_d2(6), m_d3(10), m_di{0, 0, 0, m_d1 - 1, m_d1}, m_dj{0, 0, 0, m_d2 - 1, m_d2}, m_grid(m_di, m_dj),
          m_gridb(m_di, m_dj), m_meta(m_d1, m_d2, m_d3),
          m_in(m_meta, [](int i, int j, int k) { return i + j + k; }, "in"), m_out(m_meta, -1., "out"),
          m_ref(m_meta, -1., "ref"), m_inv(make_host_view(m_in)), m_outv(make_host_view(m_out)),
          m_refv(make_host_view(m_ref)) {
        m_grid.value_list[0] = 0;
        m_grid.value_list[1] = m_d3 - 1;
        m_gridb.value_list[0] = 0;
        m_gridb.value_list[1] = m_d3 - 3;
        m_gridb.value_list[2] = m_d3 - 1;

        init_fields();
    }

    storage_t create_new_field(std::string name) { return storage_t(m_meta, -1, name); }

    void init_fields() {
        for (gridtools::uint_t i = 0; i < m_d1; ++i) {
            for (gridtools::uint_t j = 0; j < m_d2; ++j) {
                for (gridtools::uint_t k = 0; k < m_d3; ++k) {
                    m_outv(i, j, k) = -1;
                    m_refv(i, j, k) = -1;
                }
            }
        }
        m_out.clone_to_device();
    }
};
