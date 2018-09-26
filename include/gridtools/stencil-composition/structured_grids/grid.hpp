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
#include "../grid_base.hpp"

namespace gridtools {

    template <typename Axis>
    struct grid : grid_base<Axis> {
        using base_type = grid_base<Axis>;
        using grid_type = grid_type::structured;

        DEPRECATED_REASON("This constructor does not initialize the vertical axis, use the constructor with 3 "
                          "arguments. (deprecated after 1.05.02)")
        GT_FUNCTION explicit grid(halo_descriptor const &direction_i, halo_descriptor const &direction_j)
            : base_type(direction_i, direction_j) {}

        GT_FUNCTION
        explicit grid(halo_descriptor const &direction_i,
            halo_descriptor const &direction_j,
            const decltype(base_type::value_list) &value_list)
            : base_type(direction_i, direction_j, value_list) {}

        DEPRECATED_REASON("Use constructor with halo_descriptors (deprecated after 1.05.02)")
        GT_FUNCTION explicit grid(uint_t *i, uint_t *j) : base_type(i, j) {}
    };

    template <typename Grid>
    struct is_grid : boost::mpl::false_ {};

    template <typename Axis>
    struct is_grid<grid<Axis>> : boost::mpl::true_ {};

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
