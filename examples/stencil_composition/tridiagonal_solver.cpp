/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <iostream>

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
  the algorithm. This choice corresponds to having the same vector index for each row of the matrix.
 */

namespace gt = gridtools;

#ifdef __CUDACC__
using target_t = gt::target::cuda;
#else
using target_t = gt::target::mc;
#endif

using backend_t = gt::backend<target_t>;

// This is the definition of the special regions in the "vertical" direction
using axis_t = gt::axis<1>;
using full_t = axis_t::full_interval;

struct forward_thomas {
    // five vectors: output, the 3 diagonals, and the right hand side
    using out = gt::inout_accessor<0>;
    using inf = gt::in_accessor<1>;
    using diag = gt::in_accessor<2>;
    using sup = gt::inout_accessor<3>;
    using rhs = gt::inout_accessor<4>;
    using param_list = gt::make_param_list<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, 0>) {
        eval(sup{}) = eval(sup{}) / (eval(diag{}) - eval(sup{0, 0, -1}) * eval(inf{}));
        eval(rhs{}) =
            (eval(rhs{}) - eval(inf{}) * eval(rhs{0, 0, -1})) / (eval(diag{}) - eval(sup{0, 0, -1}) * eval(inf{}));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
        eval(sup{}) = eval(sup{}) / eval(diag{});
        eval(rhs{}) = eval(rhs{}) / eval(diag{});
    }
};

struct backward_thomas {
    using out = gt::inout_accessor<0>;
    using inf = gt::in_accessor<1>;
    using diag = gt::in_accessor<2>;
    using sup = gt::inout_accessor<3>;
    using rhs = gt::inout_accessor<4>;
    using param_list = gt::make_param_list<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
        eval(out{}) = eval(rhs{}) - eval(sup{}) * eval(out{0, 0, 1});
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(out{}) = eval(rhs{});
    }
};

int main() {
    unsigned int d1 = 10;
    unsigned int d2 = 10;
    unsigned int d3 = 6;

    using storage_tr = gt::storage_traits<backend_t::backend_id_t>;

    // storage_info contains the information about sizes and layout of the storages to which it will be passed
    using storage_info_t = storage_tr::storage_info_t<0, 3>;

    using storage_type = storage_tr::data_store_t<double, storage_info_t>;

    // Definition of the actual data fields that are used for input/output, instantiated using the storage_info
    auto out = storage_type{storage_info_t{d1, d2, d3}};
    auto sup = storage_type{storage_info_t{d1, d2, d3}, 1.};
    auto rhs = storage_type{storage_info_t{d1, d2, d3}, [](int, int, int k) { return k == 0 ? 4. : k == 5 ? 2. : 3.; }};

    // Definition of placeholders. The order does not have any semantics
    gt::arg<0, storage_type> p_inf;
    gt::arg<1, storage_type> p_diag;
    gt::arg<2, storage_type> p_sup;
    gt::arg<3, storage_type> p_rhs;
    gt::arg<4, storage_type> p_out;

    // Now we describe the itaration space. The first two dimensions
    // are described with a tuple of values (minus, plus, begin, end,
    // length). Begin and end, for each dimension represent the space
    // where the output data will be located in the data_stores, while
    // minus and plus indicate the number of halo points in the
    // indices before begin and after end, respectively.  In this
    // example there are no halo points needed, but distributed memory
    // applications usually have halos defined on all data fields, so
    // the halos are not only prescribed by the stencils, but also by
    // other requirements of the applications.
    gt::halo_descriptor di{0, 0, 0, d1 - 1, d1};
    gt::halo_descriptor dj{0, 0, 0, d2 - 1, d2};

    // The grid represents the iteration space. The third dimension is
    // indicated here as a size and the iteration space is deduced by
    // the fact that there is not an axis definition. More complex
    // third dimensions are possible but not described in this
    // example.
    auto grid = gt::make_grid(di, dj, d3);

    // Here we make the computation, specifying the backend, the grid
    // (iteration space), binding of the placeholders to the fields
    // that will not be modified during the computation, and then the
    // stencil structure
    auto trid_solve = gt::make_computation<backend_t>(grid,
        p_inf = storage_type{storage_info_t{d1, d2, d3}, -1.},
        p_diag = storage_type{storage_info_t{d1, d2, d3}, 3.},
        p_sup = sup,
        p_rhs = rhs,
        p_out = out,
        gt::make_multistage(gt::execute::forward(), gt::make_stage<forward_thomas>(p_out, p_inf, p_diag, p_sup, p_rhs)),
        gt::make_multistage(
            gt::execute::backward(), gt::make_stage<backward_thomas>(p_out, p_inf, p_diag, p_sup, p_rhs)));

    // Executing the computation
    trid_solve.run();

    out.sync();
    // In this simple example the solution is known and we can easily
    // check it.
    bool correct = true;
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                correct &= gt::make_host_view(out)(i, j, k) == 1.;
            }
        }
    }

    if (correct) {
        std::cout << "Passed\n";
    } else {
        std::cout << "Failed\n";
    }
}
