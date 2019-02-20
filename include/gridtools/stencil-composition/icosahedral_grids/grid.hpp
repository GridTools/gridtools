/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
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
        using grid_type_t = grid_type::icosahedral;

        GT_DEPRECATED("Use constructor with halo_descriptors (deprecated after 1.05.02)")
        GT_FUNCTION
        explicit grid(GridTopology const &grid_topology, const array<uint_t, 5> &i, const array<uint_t, 5> &j)
            : grid_base<Axis>(halo_descriptor(i[minus], i[plus], i[begin], i[end], i[length]),
                  halo_descriptor(j[minus], j[plus], j[begin], j[end], j[length])),
              m_grid_topology(grid_topology) {}

        GT_DEPRECATED("This constructor does not initialize the vertical axis, use the constructor with 4 "
                      "arguments.  (deprecated after 1.05.02)")
        GT_FUNCTION explicit grid(
            GridTopology const &grid_topology, halo_descriptor const &direction_i, halo_descriptor const &direction_j)
            : grid_base<Axis>(direction_i, direction_j), m_grid_topology(grid_topology) {}

        GT_FUNCTION
        explicit grid(GridTopology const &grid_topology,
            halo_descriptor const &direction_i,
            halo_descriptor const &direction_j,
            const decltype(grid_base<Axis>::value_list) &value_list)
            : grid_base<Axis>(direction_i, direction_j, value_list), m_grid_topology(grid_topology) {}

        GT_FUNCTION
        GridTopology const &grid_topology() const { return m_grid_topology; }
    };

    template <typename Grid>
    struct is_grid : boost::mpl::false_ {};

    template <typename Axis, typename GridTopology>
    struct is_grid<grid<Axis, GridTopology>> : boost::mpl::true_ {};

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
