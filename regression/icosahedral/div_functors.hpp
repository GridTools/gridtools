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
        GT_FUNCTION static void Do(Evaluation eval) {
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
        GT_FUNCTION static void Do(Evaluation eval) {
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
        GT_FUNCTION static void Do(Evaluation eval) {
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
