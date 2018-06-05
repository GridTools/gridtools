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

#include "backend_select.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "gtest/gtest.h"

using axis_t = gridtools::axis< 1, 1 >;
using axis = axis_t::axis_interval_t;

using kfull = axis_t::full_interval;
using kbody_high = kfull::modify< 1, 0 >;
using kminimum = kfull::first_level;
using kminimump1 = kminimum::shift< 1 >;
using kbody_highp1 = kbody_high::modify< 1, 0 >;
using kbody_highp1m1 = kbody_high::modify< 1, -1 >;
using kmaximum = kfull::last_level;
using kmaximumm1 = kmaximum::shift< -1 >;
using kbody_low = kfull::modify< 0, -1 >;
using kbody = kfull::modify< 1, -1 >;

using axis_b_t = gridtools::axis< 2, 1 >;
using axis_b = axis_b_t::axis_interval_t;

using kfull_b = axis_b_t::full_interval;
using kminimum_b = kfull_b::first_level;
using kminimump1_b = kminimum_b::shift< 1 >;
using kmaximum_b = kfull_b::last_level;
using kmaximumm1_b = axis_b_t::get_interval< 1 >::modify< 0, -1 >;
using kbody_low_b = axis_b_t::get_interval< 0 >;
using kbody_lowp1_b = kbody_low_b::modify< 1, 0 >;

class kcachef : public ::testing::Test {
  protected:
    typedef gridtools::storage_traits< backend_t::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< gridtools::float_type, storage_info_t >
        storage_t;

    const gridtools::uint_t m_d1, m_d2, m_d3;

    gridtools::halo_descriptor m_di, m_dj;

    gridtools::grid< axis > m_grid;
    gridtools::grid< axis_b > m_gridb;
    storage_info_t m_meta;
    storage_t m_in, m_out, m_ref;
    gridtools::data_view< storage_t, gridtools::access_mode::ReadWrite > m_inv, m_outv, m_refv;

    kcachef()
        : m_d1(6), m_d2(6), m_d3(10), m_di{0, 0, 0, m_d1 - 1, m_d1}, m_dj{0, 0, 0, m_d2 - 1, m_d2},
          m_grid(make_grid(m_di, m_dj, axis_t(m_d3))), m_gridb(make_grid(m_di, m_dj, axis_b_t(m_d3 - 2u, 2u))),
          m_meta(m_d1, m_d2, m_d3), m_in(m_meta, [](int i, int j, int k) { return i + j + k; }, "in"),
          m_out(m_meta, -1., "out"), m_ref(m_meta, -1., "ref"), m_inv(make_host_view(m_in)),
          m_outv(make_host_view(m_out)), m_refv(make_host_view(m_ref)) {
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
