/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <gridtools/stencil-composition/stencil-composition.hpp>

namespace ico_operators {

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

    template <uint_t Color>
    struct div_prep_functor {
        using edge_length = in_accessor<0, edges, extent<0, 1, 0, 1>>;
        using cell_area_reciprocal = in_accessor<1, cells>;
        using orientation_of_normal = in_accessor<2, cells, extent<>, 5>;
        using weights = inout_accessor<3, cells, 5>;

        using param_list = make_param_list<edge_length, cell_area_reciprocal, orientation_of_normal, weights>;

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
        using in_edges = in_accessor<0, edges, extent<0, 1, 0, 1>>;
        using weights = in_accessor<1, cells, extent<>, 5>;
        using out_cells = inout_accessor<2, cells>;

        using param_list = make_param_list<in_edges, weights, out_cells>;

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
        using in_edges = in_accessor<0, edges, extent<0, 1, 0, 1>>;
        using edge_length = in_accessor<1, edges, extent<0, 1, 0, 1>>;
        using cell_area_reciprocal = in_accessor<2, cells>;
        using out_cells = inout_accessor<3, cells>;

        using param_list = make_param_list<in_edges, edge_length, cell_area_reciprocal, out_cells>;

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
