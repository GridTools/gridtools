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

namespace gridtools {

    template <typename Axis>
    struct grid : grid_base<Axis> {
        using base_type = grid_base<Axis>;

        GT_FUNCTION
        explicit grid(halo_descriptor const &direction_i,
            halo_descriptor const &direction_j,
            const decltype(base_type::value_list) &value_list)
            : base_type(direction_i, direction_j, value_list) {}
    };

    template <class T>
    GT_META_DEFINE_ALIAS(is_grid, meta::is_instantiation_of, (grid, T));

    template <typename Axis>
    GT_FUNCTION_HOST grid<typename Axis::axis_interval_t> make_grid(
        halo_descriptor const &direction_i, halo_descriptor const &direction_j, Axis axis) {
        return grid<typename Axis::axis_interval_t>(
            direction_i, direction_j, _impl::intervals_to_indices(axis.interval_sizes()));
    }
    GT_FUNCTION_HOST grid<axis<1>::axis_interval_t> make_grid(uint_t di, uint_t dj, uint_t dk) {
        return make_grid(halo_descriptor(di), halo_descriptor(dj), axis<1>(dk));
    }
    template <typename Axis>
    GT_FUNCTION_HOST grid<typename Axis::axis_interval_t> make_grid(uint_t di, uint_t dj, Axis axis) {
        return grid<typename Axis::axis_interval_t>(
            halo_descriptor(di), halo_descriptor(dj), _impl::intervals_to_indices(axis.interval_sizes()));
    }
    GT_FUNCTION_HOST grid<axis<1>::axis_interval_t> make_grid(
        halo_descriptor const &direction_i, halo_descriptor const &direction_j, uint_t dk) {
        return make_grid(direction_i, direction_j, axis<1>(dk));
    }
} // namespace gridtools
