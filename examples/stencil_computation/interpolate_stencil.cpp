/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "interpolate_stencil.hpp"

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/stencil_composition/global_parameter.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef GT_CUDACC
#include <gridtools/stencil_composition/backend/cuda.hpp>
using backend_t = gridtools::cuda::backend<>;
#else
#include <gridtools/stencil_composition/backend/mc.hpp>
using backend_t = gridtools::mc::backend;
#endif

namespace {
    using namespace gridtools;
    using namespace cartesian;

    struct interpolate_stage {
        using in1 = in_accessor<0>;
        using in2 = in_accessor<1>;
        using weight = in_accessor<2>;
        using out = inout_accessor<3>;

        using param_list = make_param_list<in1, in2, weight, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            using namespace expressions;
            eval(out()) = eval(weight() * in1() + (1. - weight()) * in2());
        }
    };
} // namespace

// `run_single_stage` should never be called in a header, because the compilation overhead is very significant
std::function<void(inputs, outputs)> make_interpolate_stencil(grid_t grid, double weight) {
    return [grid = std::move(grid), weight](inputs in, outputs out) {
        run_single_stage(
            interpolate_stage(), backend_t(), grid, in.in1, in.in2, make_global_parameter(weight), out.out);
    };
}
