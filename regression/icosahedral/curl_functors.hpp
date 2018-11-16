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
    struct curl_prep_functor {
        using dual_area_reciprocal = in_accessor<0, vertices>;
        using dual_edge_length = in_accessor<1, edges, extent<-1, 0, -1, 0>>;
        using weights = inout_accessor<2, vertices, 5>;
        using edge_orientation = in_accessor<3, vertices, extent<>, 5>;

        using arg_list = boost::mpl::vector<dual_area_reciprocal, dual_edge_length, weights, edge_orientation>;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation eval) {
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
        using in_edges = in_accessor<0, edges, extent<-1, 0, -1, 0>>;
        using weights = in_accessor<1, vertices, extent<>, 5>;
        using out_vertices = inout_accessor<2, vertices>;

        using arg_list = boost::mpl::vector<in_edges, weights, out_vertices>;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation eval) {
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
        using in_edges = in_accessor<0, edges, extent<-1, 0, -1, 0>>;
        using dual_area_reciprocal = in_accessor<1, vertices>;
        using dual_edge_length = in_accessor<2, edges, extent<-1, 0, -1, 0>>;
        using out_vertices = inout_accessor<3, vertices>;

        using arg_list = boost::mpl::vector<in_edges, dual_area_reciprocal, dual_edge_length, out_vertices>;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation eval) {
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
