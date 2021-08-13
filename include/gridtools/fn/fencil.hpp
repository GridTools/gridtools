/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <tuple>

namespace gridtools::fn {
    template <class Domain, class Stencil, class Outputs, class... Inputs>
    struct closure {
        Domain domain;
        Stencil stencil;
        Outputs outputs;
        std::tuple<Inputs const &...> inputs;

        closure(Domain domain, Stencil stencil, Outputs outputs, Inputs const &... inputs)
            : domain(domain), stencil(stencil), outputs(outputs), inputs(inputs...) {}
    };

    auto out(auto &... args) { return std::tie(args...); }

    void fencil(auto const &backend, auto &&... closures) {
        (..., (fn_apply(backend, closures.domain, closures.stencil, closures.outputs, closures.inputs)));
    }
} // namespace gridtools::fn
