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

#include "../../meta.hpp"
#include "../grid_base.hpp"
#include "./icosahedral_topology.hpp"

namespace gridtools {

    template <typename Axis, typename GridTopology>
    struct grid : grid_base<Axis> {
        GT_STATIC_ASSERT((is_interval<Axis>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid_topology<GridTopology>::value), GT_INTERNAL_ERROR);

        typedef GridTopology grid_topology_t;

      private:
        GridTopology m_grid_topology;

      public:
        GT_FUNCTION
        explicit grid(GridTopology const &grid_topology,
            halo_descriptor const &direction_i,
            halo_descriptor const &direction_j,
            const decltype(grid_base<Axis>::value_list) &value_list)
            : grid_base<Axis>(direction_i, direction_j, value_list), m_grid_topology(grid_topology) {}

        GT_FUNCTION
        GridTopology const &grid_topology() const { return m_grid_topology; }
    };

    template <class T>
    using is_grid = meta::is_instantiation_of<grid, T>;

    template <typename Axis, typename GridTopology>
    GT_FUNCTION_HOST grid<typename Axis::axis_interval_t, GridTopology> make_grid(GridTopology const &grid_topology,
        halo_descriptor const &direction_i,
        halo_descriptor const &direction_j,
        Axis axis) {
        return grid<typename Axis::axis_interval_t, GridTopology>(
            grid_topology, direction_i, direction_j, _impl::intervals_to_indices(axis.interval_sizes()));
    }
    template <typename GridTopology>
    GT_FUNCTION_HOST grid<axis<1>::axis_interval_t, GridTopology> make_grid(
        GridTopology const &grid_topology, uint_t di, uint_t dj, uint_t dk) {
        return make_grid(grid_topology, halo_descriptor(di), halo_descriptor(dj), axis<1>(dk));
    }
    template <typename Axis, typename GridTopology>
    GT_FUNCTION_HOST grid<typename Axis::axis_interval_t, GridTopology> make_grid(
        GridTopology const &grid_topology, uint_t di, uint_t dj, Axis axis) {
        return grid<typename Axis::axis_interval_t, GridTopology>(grid_topology,
            halo_descriptor(di),
            halo_descriptor(dj),
            _impl::intervals_to_indices(axis.interval_sizes()));
    }
    template <typename GridTopology>
    GT_FUNCTION_HOST grid<axis<1>::axis_interval_t, GridTopology> make_grid(GridTopology const &grid_topology,
        halo_descriptor const &direction_i,
        halo_descriptor const &direction_j,
        uint_t dk) {
        return make_grid(grid_topology, direction_i, direction_j, axis<1>(dk));
    }
} // namespace gridtools
