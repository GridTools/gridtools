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

#include <gridtools/stencil_composition/stencil_composition.hpp>

namespace ico_operators {

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

    template <uint_t Color>
    struct div_prep_functor {
        GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(edge_length, edges, extent<0, 1, 0, 1>),
            GT_IN_ACCESSOR(cell_area_reciprocal, cells),
            GT_IN_ACCESSOR(orientation_of_normal, cells, extent<>, 5),
            GT_INOUT_ACCESSOR(weights, cells, 5));

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            constexpr dimension<5> edge;
            constexpr auto neighbors_offsets = connectivity<cells, edges, Color>::offsets();
            int_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(weights(edge + e)) = eval(orientation_of_normal(edge + e)) * eval(edge_length(neighbor_offset)) *
                                          eval(cell_area_reciprocal());
                e++;
            }
        }
    };

    template <uint_t Color>
    struct div_functor_reduction_into_scalar {
        GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(in_edges, edges, extent<0, 1, 0, 1>),
            GT_IN_ACCESSOR(weights, cells, extent<>, 5),
            GT_INOUT_ACCESSOR(out_cells, cells));

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            constexpr auto neighbors_offsets = connectivity<cells, edges, Color>::offsets();
            constexpr dimension<5> edge;

            double t = 0;
            int_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                t += eval(in_edges(neighbor_offset)) * eval(weights(edge + e));
                e++;
            }
            eval(out_cells()) = t;
        }
    };

    template <uint_t Color>
    struct div_functor_flow_convention_connectivity {
        GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(in_edges, edges, extent<0, 1, 0, 1>),
            GT_IN_ACCESSOR(edge_length, edges, extent<0, 1, 0, 1>),
            GT_IN_ACCESSOR(cell_area_reciprocal, cells),
            GT_INOUT_ACCESSOR(out_cells, cells));

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            constexpr auto neighbors_offsets = connectivity<cells, edges, Color>::offsets();
            double t = 0;
            for (auto neighbor_offset : neighbors_offsets)
                t += eval(in_edges(neighbor_offset)) * eval(edge_length(neighbor_offset));

            if (Color == 0)
                eval(out_cells()) = t * eval(cell_area_reciprocal());
            else
                eval(out_cells()) = -t * eval(cell_area_reciprocal());
        }
    };
} // namespace ico_operators
