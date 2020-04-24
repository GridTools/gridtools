/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
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

#include <iostream>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/stencil_composition/global_parameter.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef GT_CUDACC
#include <gridtools/stencil_composition/backend/cuda.hpp>
#include <gridtools/storage/cuda.hpp>
using backend_t = gridtools::cuda::backend<>;
using storage_traits_t = gridtools::storage::cuda;
#else
#include <gridtools/stencil_composition/backend/mc.hpp>
#include <gridtools/storage/mc.hpp>
using backend_t = gridtools::mc::backend<>;
using storage_traits_t = gridtools::storage::mc;
#endif

namespace gt = gridtools;

// This is the definition of the special regions in the "vertical" direction
using full_t = gt::axis<1>::full_interval;

struct forward_thomas {
    // five vectors: output, the 3 diagonals, and the right hand side
    using out = gt::cartesian::inout_accessor<0>;
    using inf = gt::cartesian::in_accessor<1>;
    using diag = gt::cartesian::in_accessor<2>;
    using sup = gt::cartesian::inout_accessor<3, gt::extent<0, 0, 0, 0, -1, 0>>;
    using rhs = gt::cartesian::inout_accessor<4, gt::extent<0, 0, 0, 0, -1, 0>>;
    using param_list = gt::make_param_list<out, inf, diag, sup, rhs>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, full_t::modify<1, 0>) {
        eval(sup{}) = eval(sup{}) / (eval(diag{}) - eval(sup{0, 0, -1}) * eval(inf{}));
        eval(rhs{}) =
            (eval(rhs{}) - eval(inf{}) * eval(rhs{0, 0, -1})) / (eval(diag{}) - eval(sup{0, 0, -1}) * eval(inf{}));
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, full_t::first_level) {
        eval(sup{}) = eval(sup{}) / eval(diag{});
        eval(rhs{}) = eval(rhs{}) / eval(diag{});
    }
};

struct backward_thomas {
    using out = gt::cartesian::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 1>>;
    using inf = gt::cartesian::in_accessor<1>;
    using diag = gt::cartesian::in_accessor<2>;
    using sup = gt::cartesian::inout_accessor<3>;
    using rhs = gt::cartesian::inout_accessor<4>;
    using param_list = gt::make_param_list<out, inf, diag, sup, rhs>;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, full_t::modify<0, -1>) {
        eval(out{}) = eval(rhs{}) - eval(sup{}) * eval(out{0, 0, 1});
    }

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval, full_t::last_level) {
        eval(out{}) = eval(rhs{});
    }
};

int main() {
    unsigned d1 = 10;
    unsigned d2 = 10;
    unsigned d3 = 6;

    // Add dimensions and the field type to the storage builder
    auto storage_builder = gt::storage::builder<storage_traits_t>.dimensions(d1, d2, d3).type<double>();

    // Definition of the actual data fields that are used for input/output
    auto out = storage_builder.build();
    auto sup = storage_builder.value(1).build();
    auto rhs = storage_builder.initializer([](int, int, int k) { return k == 0 ? 4 : k == 5 ? 2 : 3; }).build();

    // The grid represents the iteration space. The third dimension is indicated here as a size and the iteration space
    // is deduced by the fact that there is not an axis definition. More complex third dimensions are possible but not
    // described in this example.
    auto grid = gt::make_grid(d1, d2, d3);

    auto spec = [](auto inf, auto diag, auto sup, auto rhs, auto out) {
        return gt::multi_pass(gt::execute_forward().stage(forward_thomas(), out, inf, diag, sup, rhs),
            gt::execute_backward().stage(backward_thomas(), out, inf, diag, sup, rhs));
    };

    // Here we make the computation, specifying the backend, the grid (iteration space), binding of the spec arguments
    // to the fields
    gt::run(spec, backend_t(), grid, gt::make_global_parameter(-1), gt::make_global_parameter(3), sup, rhs, out);

    // In this simple example the solution is known and we can easily check it.
    auto view = out->const_host_view();
    for (unsigned i = 0; i < d1; ++i)
        for (unsigned j = 0; j < d2; ++j)
            for (unsigned k = 0; k < d3; ++k)
                if (std::abs(view(i, j, k) - 1) > 1e-10) {
                    std::cerr << "Failure " << view(i, j, k) - 1 << std::endl;
                    return 1;
                }
    std::cout << "Success" << std::endl;
}
