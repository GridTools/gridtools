/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using axis_t = gridtools::axis<3, 1, 3>;
using axis = axis_t::axis_interval_t;

using kfull = axis_t::full_interval;
using kbody = kfull::modify<1, -1>;
using kminimum = kfull::first_level;
using kminimumm1 = kminimum::shift<-1>;
using kminimump1 = kminimum::shift<1>;
using kmaximum = kfull::last_level;
using kmaximumm1 = kmaximum::shift<-1>;
using kmaximump1 = kmaximum::shift<1>;
using kmaximum2 = axis_t::get_interval<2>;
using kmaximum_m2 = axis_t::get_interval<1>::last_level;
using kmaximum_m3 = kmaximum_m2::shift<-1>;

using kbody_high = kfull::modify<1, 0>;
using kbody_highp1 = kbody_high::modify<1, 0>;

using kbody_low = kfull::modify<0, -1>;
using kbody_low_m1 = kbody_low::modify<0, -1>;
using kbody_lowp1 = kbody_low_m1::modify<1, 0>;
using kbody_highp1m1 = kbody_highp1::modify<0, -1>;

using lasttwo = axis_t::get_interval<2>;
using midbody = axis_t::get_interval<1>;
using midbody_last = midbody::last_level;
using midbody_first = midbody::first_level;
using midbody_low = midbody::modify<0, -1>;
using midbody_high = midbody::modify<1, 0>;
using firsttwo = axis_t::get_interval<0>;
using fullminustwolast = midbody::modify<-2, 0>;
using fullminustwofirst = midbody::modify<0, 2>;

class kcachef : public ::testing::Test {
  protected:
    typedef gridtools::storage_traits<backend_t::target_t>::storage_info_t<0, 3> storage_info_t;
    typedef gridtools::storage_traits<backend_t::target_t>::data_store_t<float_type, storage_info_t> storage_t;

    const gridtools::uint_t m_d1, m_d2, m_d3;

    gridtools::halo_descriptor m_di, m_dj;

    gridtools::grid<axis> m_grid;

    storage_info_t m_meta;
    storage_t m_in, m_out, m_ref;
    gridtools::data_view<storage_t, gridtools::access_mode::read_write> m_inv, m_outv, m_refv;

    kcachef()
        : m_d1(6), m_d2(6), m_d3(10), m_di{0, 0, 0, m_d1 - 1, m_d1}, m_dj{0, 0, 0, m_d2 - 1, m_d2},
          m_grid(make_grid(m_di, m_dj, axis_t(2u, m_d3 - 4u, 2u))), m_meta(m_d1, m_d2, m_d3),
          m_in(m_meta, [](int i, int j, int k) { return i + j + k + 1; }, "in"), m_out(m_meta, -1., "out"),
          m_ref(m_meta, -1., "ref"), m_inv(make_host_view(m_in)), m_outv(make_host_view(m_out)),
          m_refv(make_host_view(m_ref)) {
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
