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

#include <utility>

#include <gtest/gtest.h>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/halo_descriptor.hpp"
#include "../common/selector.hpp"
#include "../stencil_composition/axis.hpp"
#include "../stencil_composition/grid.hpp"
#include "../storage/common/halo.hpp"
#include "../storage/storage_facility.hpp"
#include "backend_select.hpp"
#include "verifier.hpp"

namespace gridtools {
    template <size_t HaloSize = 0, class Axis = axis<1>>
    class computation_fixture : virtual public ::testing::Test {
        uint_t m_d1;
        uint_t m_d2;
        uint_t m_d3;

        template <class StorageType>
        using halos_t = array<array<uint_t, 2>, StorageType::storage_info_t::layout_t::masked_length>;

      public:
        static constexpr uint_t halo_size = HaloSize;
        using storage_tr = storage_traits<backend_t>;

        halo_descriptor i_halo_descriptor() const {
            return {halo_size, halo_size, halo_size, m_d1 - halo_size - 1, m_d1};
        }
        halo_descriptor j_halo_descriptor() const {
            return {halo_size, halo_size, halo_size, m_d2 - halo_size - 1, m_d2};
        }

        auto make_grid() const { return ::gridtools::make_grid(i_halo_descriptor(), j_halo_descriptor(), Axis{m_d3}); }

#ifndef GT_ICOSAHEDRAL_GRIDS
        using halo_t = halo<HaloSize, HaloSize, 0>;
        using storage_info_t = storage_tr::storage_info_t<0, 3, halo_t>;
        using j_storage_info_t = storage_tr::special_storage_info_t<1, selector<0, 1, 0>>;

        using storage_type = storage_tr::data_store_t<float_type, storage_info_t>;
        using j_storage_type = storage_tr::data_store_t<float_type, j_storage_info_t>;

        template <class Storage = storage_type, class T = typename Storage::data_t>
        Storage make_storage(T &&obj = {}) const {
            return {{m_d1, m_d2, m_d3}, std::forward<T>(obj)};
        }

        template <class Storage = storage_type>
        Storage make_storage(double val) const {
            return {{m_d1, m_d2, m_d3}, (typename Storage::data_t)val};
        }

      private:
        template <class Storage>
        static halos_t<Storage> halos() {
            return {{ {halo_size, halo_size}, { halo_size, halo_size } }};
        }

      public:
#else
        using cells = enumtype::cells;
        using edges = enumtype::edges;
        using vertices = enumtype::vertices;

        using halo_t = halo<HaloSize, 0, HaloSize, 0>;

        template <class Location, class Selector = selector<1, 1, 1, 1>, class Halo = halo_t>
        using storage_info_t = icosahedral_storage_info_type<backend_t, Location, Halo, Selector>;

        template <class Location, class Selector = selector<1, 1, 1, 1>, class Halo = halo_t>
        using storage_type = storage_tr::data_store_t<float_type, storage_info_t<Location, Selector, Halo>>;

        using edge_2d_storage_type = storage_type<edges, selector<1, 1, 1, 0>>;
        using cell_2d_storage_type = storage_type<cells, selector<1, 1, 1, 0>>;
        using vertex_2d_storage_type = storage_type<vertices, selector<1, 1, 1, 0>>;

        template <class Location, class Selector = selector<1, 1, 1, 1, 1>>
        using storage_type_4d = storage_type<Location, Selector, halo<halo_size, 0, halo_size, 0, 0>>;

        template <uint_t I, class Location>
        using arg = gridtools::arg<I, void, Location>;

        template <uint_t I, class Location, typename Storage = float_type>
        using tmp_arg = gridtools::tmp_arg<I, Storage, Location>;

        template <class Location, class Storage = storage_type<Location>, class T = typename Storage::data_t>
        Storage make_storage(T &&obj = {}) const {
            return {{m_d1, Location::n_colors::value, m_d2, m_d3}, std::forward<T>(obj)};
        }

        template <class Location, class Storage = storage_type<Location>>
        Storage make_storage(double val) const {
            return {{m_d1, Location::n_colors::value, m_d2, m_d3}, (typename Storage::data_t)val};
        }

        template <class Location, class Storage = storage_type_4d<Location>, class T = typename Storage::data_t>
        Storage make_storage_4d(uint_t dim, T &&val = {}) {
            return {{d1(), Location::n_colors::value, d2(), d3(), dim}, std::forward<T>(val)};
        }

        template <class Location, class Storage = storage_type_4d<Location>>
        Storage make_storage_4d(uint_t dim, double val) {
            return {{d1(), Location::n_colors::value, d2(), d3(), dim}, (typename Storage::data_t)val};
        }

      private:
        template <class Storage>
        static halos_t<Storage> halos() {
            return {{{halo_size, halo_size}, {}, {halo_size, halo_size}}};
        }

      public:
#endif
        /// Fixture constructor takes the dimensions of the computation
        computation_fixture(uint_t d1, uint_t d2, uint_t d3) : m_d1(d1), m_d2(d2), m_d3(d3) {}

        uint_t d1() const { return m_d1; }
        uint_t d2() const { return m_d2; }
        uint_t d3() const { return m_d3; }

        uint_t &d1() { return m_d1; }
        uint_t &d2() { return m_d2; }
        uint_t &d3() { return m_d3; }

        template <class Expected, class Actual>
        void verify(
            Expected const &expected, Actual const &actual, double precision = default_precision<float_type>()) const {
            EXPECT_TRUE(verifier{precision}.verify(make_grid(), expected, actual, halos<Expected>()));
        }
    };
} // namespace gridtools
