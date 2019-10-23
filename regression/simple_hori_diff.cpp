/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "horizontal_diffusion_repository.hpp"

/**
  @file
  This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO
 */

using namespace gridtools;

struct wlap_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;
    using crlato = in_accessor<2>;
    using crlatu = in_accessor<3>;

    using param_list = make_param_list<out, in, crlato, crlatu>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(in(1, 0)) + eval(in(-1, 0)) - float_type{2} * eval(in()) +
                      eval(crlato()) * (eval(in(0, 1)) - eval(in())) + eval(crlatu()) * (eval(in(0, -1)) - eval(in()));
    }
};

struct divflux_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1>;
    using lap = in_accessor<2, extent<-1, 1, -1, 1>>;
    using crlato = in_accessor<3>;
    using coeff = in_accessor<4>;

    using param_list = make_param_list<out, in, lap, crlato, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        auto fluxx = eval(lap(1, 0)) - eval(lap());
        auto fluxx_m = eval(lap()) - eval(lap(-1, 0));

        auto fluxy = eval(crlato()) * (eval(lap(0, 1)) - eval(lap()));
        auto fluxy_m = eval(crlato()) * (eval(lap()) - eval(lap(0, -1)));

        eval(out()) = eval(in()) + ((fluxx_m - fluxx) + (fluxy_m - fluxy)) * eval(coeff());
    }
};

const auto hori_diff = [](auto coeff, auto in, auto out, auto crlato, auto crlatu) {
    GT_DECLARE_TMP(float_type, lap);
    return execute_parallel()
        .ij_cached(lap)
        .stage(wlap_function(), lap, in, crlato, crlatu)
        .stage(divflux_function(), out, in, lap, crlato, coeff);
};

using simple_hori_diff = regression_fixture<2>;

TEST_F(simple_hori_diff, test) {
    auto out = make_storage();

    horizontal_diffusion_repository repo(d1(), d2(), d3());

    auto comp = [grid = make_grid(),
                    coeff = make_storage(repo.coeff),
                    in = make_storage(repo.in),
                    &out,
                    crlato = make_storage<j_storage_type>(repo.crlato),
                    crlatu = make_storage<j_storage_type>(repo.crlatu)] {
        run(hori_diff, backend_t(), grid, coeff, in, out, crlato, crlatu);
    };
    comp();
    verify(make_storage(repo.out_simple), out);
    benchmark(comp);
}
