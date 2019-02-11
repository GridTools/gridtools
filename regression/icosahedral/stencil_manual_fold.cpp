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

/*
 * This example demonstrates how to retrieve the connectivity information of the
 * icosahedral/octahedral grid in the user functor. This is useful for example when
 * we need to operate on fields with a double location, for which the on_cells, on_edges
 * syntax has limitations, as it requires make use of the eval object, which is not
 * resolved in the lambdas passed to the on_cells syntax.
 * The example shown here computes a value for each edge of a cells. Therefore the primary
 * location type of the output field is cells, however we do not store a scalar value, but
 * a value per edge of each cell (i.e. 3 values).
 * The storage is therefore a 5 dimensional field with indices (i, c, j, k, edge_number)
 * where the last has the range [0,2]
 *
 */

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "neighbours_of.hpp"

using namespace gridtools;
using namespace expressions;

template <uint_t Color>
struct test_on_edges_functor {
    using cell_area = in_accessor<0, enumtype::cells, extent<1>>;
    using weight_edges = inout_accessor<1, enumtype::cells, 5>;
    using param_list = make_param_list<cell_area, weight_edges>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        constexpr dimension<5> edge = {};

        // retrieve the array of neighbor offsets. This is an array with length 3 (number of neighbors).
        constexpr auto neighbors_offsets = connectivity<enumtype::cells, enumtype::cells, Color>::offsets();
        ushort_t e = 0;
        // loop over all neighbours. Each iterator (neighbor_offset) is a position offset, i.e. an array with length 4
        for (auto neighbor_offset : neighbors_offsets) {
            eval(weight_edges(edge + e)) = eval(cell_area(neighbor_offset)) / eval(cell_area());
            e++;
        }
    }
};

using stencil_manual_fold = regression_fixture<1>;

TEST_F(stencil_manual_fold, test) {
    auto in = [](int_t i, int_t c, int_t j, int_t k) { return 1. + i + c + j + k; };
    auto ref = [&](int_t i, int_t c, int_t j, int_t k, int_t e) {
        return neighbours_of<cells, cells>(i, c, j, k)[e].call(in) / in(i, c, j, k);
    };
    auto weight_edges = make_storage_4d<cells>(3);

    arg<0, cells> p_in;
    arg<1, cells, storage_type_4d<cells>> p_out;

    auto comp = make_computation(p_in = make_storage<cells>(in),
        p_out = weight_edges,
        make_multistage(execute::forward(), make_stage<test_on_edges_functor, topology_t, cells>(p_in, p_out)));

    comp.run();
    verify(make_storage_4d<cells>(3, ref), weight_edges);

    benchmark(comp);
}
