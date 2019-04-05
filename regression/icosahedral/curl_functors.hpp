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
    struct curl_prep_functor {
        GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(dual_area_reciprocal, vertices),
            GT_IN_ACCESSOR(dual_edge_length, edges, extent<-1, 0, -1, 0>),
            GT_INOUT_ACCESSOR(weights, vertices, 5),
            GT_IN_ACCESSOR(edge_orientation, vertices, extent<>, 5));

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            constexpr dimension<5> edge;
            constexpr auto neighbors_offsets = connectivity<vertices, edges, Color>::offsets();
            int_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(weights(edge + e)) = eval(edge_orientation(edge + e)) * eval(dual_edge_length(neighbor_offset)) *
                                          eval(dual_area_reciprocal());
                e++;
            }
        }
    };

    template <uint_t Color>
    struct curl_functor_weights {
        GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(in_edges, edges, extent<-1, 0, -1, 0>),
            GT_IN_ACCESSOR(weights, vertices, extent<>, 5),
            GT_INOUT_ACCESSOR(out_vertices, vertices));

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            constexpr dimension<5> edge;
            constexpr auto neighbors_offsets = connectivity<vertices, edges, Color>::offsets();
            float_type t = 0;
            int_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                t += eval(in_edges(neighbor_offset)) * eval(weights(edge + e));
                e++;
            }
            eval(out_vertices()) = t;
        }
    };

    template <uint_t Color>
    struct curl_functor_flow_convention {
        GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(in_edges, edges, extent<-1, 0, -1, 0>),
            GT_IN_ACCESSOR(dual_area_reciprocal, vertices),
            GT_IN_ACCESSOR(dual_edge_length, edges, extent<-1, 0, -1, 0>),
            GT_INOUT_ACCESSOR(out_vertices, vertices));

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            constexpr auto neighbor_offsets = connectivity<vertices, edges, Color>::offsets();
            eval(out_vertices()) = -eval(in_edges(neighbor_offsets[0])) * eval(dual_edge_length(neighbor_offsets[0])) +
                                   eval(in_edges(neighbor_offsets[1])) * eval(dual_edge_length(neighbor_offsets[1])) -
                                   eval(in_edges(neighbor_offsets[2])) * eval(dual_edge_length(neighbor_offsets[2])) +
                                   eval(in_edges(neighbor_offsets[3])) * eval(dual_edge_length(neighbor_offsets[3])) -
                                   eval(in_edges(neighbor_offsets[4])) * eval(dual_edge_length(neighbor_offsets[4])) +
                                   eval(in_edges(neighbor_offsets[5])) * eval(dual_edge_length(neighbor_offsets[5]));

            eval(out_vertices()) *= eval(dual_area_reciprocal());
        }
    };
} // namespace ico_operators
