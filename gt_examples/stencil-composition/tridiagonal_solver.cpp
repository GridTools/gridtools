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

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

/*
  @file This file shows an implementation of the Thomas algorithm, done using stencil operations.

  Important convention: the linear system as usual is represented with 4 vectors: the main diagonal
  (diag), the upper and lower first diagonals (sup and inf respectively), and the right hand side
  (rhs). Note that the dimensions and the memory layout are, for an NxN system
  rank(diag)=N       [xxxxxxxxxxxxxxxxxxxxxxxx]
  rank(inf)=N-1      [0xxxxxxxxxxxxxxxxxxxxxxx]
  rank(sup)=N-1      [xxxxxxxxxxxxxxxxxxxxxxx0]
  rank(rhs)=N        [xxxxxxxxxxxxxxxxxxxxxxxx]
  where x denotes any number and 0 denotes the padding, a dummy value which is not used in
  the algorithm. This choice coresponds to having the same vector index for each row of the matrix.
 */

namespace gt = gridtools;

// This is the definition of the special regions in the "vertical" direction
using axis_t = gt::axis<1>;
using full_t = axis_t::full_interval;

struct forward_thomas {
    // four vectors: output, and the 3 diagonals
    using out = gt::inout_accessor<0>;
    using inf = gt::in_accessor<1>;    // a
    using diag = gt::in_accessor<2>;   // b
    using sup = gt::inout_accessor<3>; // c
    using rhs = gt::inout_accessor<4>; // d
    using arg_list = gt::make_arg_list<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, full_t::modify<1, 0>) {
        eval(sup{}) = eval(sup{}) / (eval(diag{}) - eval(sup{0, 0, -1}) * eval(inf{}));
        eval(rhs{}) =
            (eval(rhs{}) - eval(inf{}) * eval(rhs{0, 0, -1})) / (eval(diag{}) - eval(sup{0, 0, -1}) * eval(inf{}));
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, full_t::first_level) {
        eval(sup{}) = eval(sup{}) / eval(diag{});
        eval(rhs{}) = eval(rhs{}) / eval(diag{});
    }
};

struct backward_thomas {
    using out = gt::inout_accessor<0>;
    using inf = gt::in_accessor<1>;    // a
    using diag = gt::in_accessor<2>;   // b
    using sup = gt::inout_accessor<3>; // c
    using rhs = gt::inout_accessor<4>; // d
    using arg_list = gt::make_arg_list<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, full_t::modify<0, 1>) {
        eval(out{}) = eval(rhs{}) - eval(sup{}) * eval(out{0, 0, 1});
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, full_t::last_level) {
        eval(out{}) = eval(rhs{});
    }
};

int main() {
    unsigned int d1 = 10;
    unsigned int d2 = 10;
    unsigned int d3 = 6;

    using storage_tr = gt::storage_traits<backend_t::backend_id_t>;

    using storage_info_t = storage_tr::storage_info_t<0, 3>;

    using storage_type = storage_tr::data_store_t<float_type, storage_info_t>;

    auto out = storage_type{storage_info_t{d1, d2, d3}};
    auto sup = storage_type{storage_info_t{d1, d2, d3}, 1.};
    auto rhs = storage_type{storage_info_t{d1, d2, d3}, [](int, int, int k) { return k == 0 ? 4. : k == 5 ? 2. : 3.; }};

    gt::arg<0, storage_type> p_inf;  // a
    gt::arg<1, storage_type> p_diag; // b
    gt::arg<2, storage_type> p_sup;  // c
    gt::arg<3, storage_type> p_rhs;  // d
    gt::arg<4, storage_type> p_out;

    gt::halo_descriptor di{0, 0, 0, d1 - 1, d1};
    gt::halo_descriptor dj{0, 0, 0, d2 - 1, d2};

    auto grid = gt::make_grid(di, dj, d3);

    auto trid_solve = gt::make_computation<backend_t>(grid,
        p_inf = storage_type{storage_info_t{d1, d2, d3}, -1.},
        p_diag = storage_type{storage_info_t{d1, d2, d3}, 3.},
        p_sup = sup,
        p_rhs = rhs,
        p_out = out,
        gt::make_multistage(gt::enumtype::execute<gt::enumtype::forward>(),
            gt::make_stage<forward_thomas>(p_out, p_inf, p_diag, p_sup, p_rhs)),
        gt::make_multistage(gt::enumtype::execute<gt::enumtype::backward>(),
            gt::make_stage<backward_thomas>(p_out, p_inf, p_diag, p_sup, p_rhs)));

    trid_solve.run();

    //    verify(make_storage(1.), out);
}
