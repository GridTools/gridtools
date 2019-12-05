/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/meta.hpp>
#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/stencil_composition/global_parameter.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;
using namespace cartesian;
using namespace expressions;

/**
  @file
  @brief This file shows a possible usage of the extension to storages with more than 3 space dimensions.

  We recall that the space dimensions simply identify the number of indexes/strides required to access
  a contiguous chunk of storage. The number of space dimensions is fully arbitrary.

  In particular, we show how to perform a nested inner loop on the extra dimension(s). Possible scenarios
  where this can be useful could be:
  * when dealing with arbitrary order integration of a field in the cells.
  * when we want to implement a discretization scheme involving integrals (like all Galerkin-type discretizations, i.e.
  continuous/discontinuous finite elements, isogeometric analysis)
  * if we discretize an equation defined on a manifold with more than 3 dimensions (e.g. space-time)
  * if we want to implement coloring schemes, or access the grid points using exotic (but 'regular') patterns

  In this example we suppose that we aim at projecting a field 'f' on a finite elements space. To each
  i,j,k point corresponds an element (we can e.g. suppose that the i,j,k, nodes are the low-left corner).
  We suppose that the following (4-dimensional) quantities are provided (replaced with stubs)
  * The basis and test functions phi and psi respectively, evaluated on the quadrature points of the
  reference element
  * The Jacobian of the finite elements transformation (from the reference to the current configurations)
  , also evaluated in the quadrature points
  * The quadrature nodes/quadrature rule

  With this information we perform the projection (i.e. perform an integral) by looping on the
  quadrature points in an innermost loop, with stride given by the layout_map (I*J*K in this case).

  Note that the fields phi and psi are passed through as global_parameters and taken in the stencil
  operator as global_accessors. This is the czse since the base functions do not change when the
  iteration point moves, so their values are constant. This is a typical example of global_parameter use.
*/

struct integration {
    using phi_fun = in_accessor<0>;
    using psi_fun = in_accessor<1>;
    using jac = in_accessor<2, extent<>, 4>;
    using f = in_accessor<3, extent<>, 6>;
    using result = inout_accessor<4, extent<>, 6>;

    using param_list = make_param_list<phi_fun, psi_fun, jac, f, result>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        dimension<4> di;
        dimension<5> dj;
        dimension<6> dk;
        dimension<4> qp;
        auto psi = eval(psi_fun());
        auto phi = eval(phi_fun());
        using float_type = std::decay_t<decltype(eval(result()))>;
        for (int I = 0; I < 2; ++I)
            for (int J = 0; J < 2; ++J)
                for (int K = 0; K < 2; ++K) {
                    float_type res = 0;
                    for (int q = 0; q < 2; ++q) {
                        float_type sum = 0;
                        for (int a = 0; a < 2; ++a)
                            for (int b = 0; b < 2; ++b)
                                for (int c = 0; c < 2; ++c)
                                    sum += eval(psi(a, b, c, q) * f(di + a, dj + b, dk + c));
                        res += eval(phi(I, J, K, q) * jac(qp + q) * sum);
                    }
                    eval(result(di + I, dj + J, dk + K)) = res / 8;
                }
    }
};

using extended_4d = regression_fixture<>;

/**
 * this is a user-defined class which will be used from within the user functor
 * by calling its  operator(). It can represent in this case values which are local to the elements
 * e.g. values of the basis functions in the quad points.
 */
struct elemental {
    GT_FUNCTION double operator()(int, int, int, int) const { return m_val; }

    double m_val;
};

TEST_F(extended_4d, test) {
    static constexpr uint_t nbQuadPt = 2;
    static constexpr uint_t b1 = 2;
    static constexpr uint_t b2 = 2;
    static constexpr uint_t b3 = 2;

    float_type phi = 10, psi = 11, f = 1.3;
    auto jac = [](int, int, int, int q) { return 1 + q; };
    const auto const_builder = storage::builder<storage_traits_t>.type<float_type const>();

    auto result = storage::builder<storage_traits_t>.type<float_type>().dimensions(d1(), d2(), d3(), b1, b2, b3)();
    easy_run(integration(),
        backend_t(),
        make_grid(),
        make_global_parameter(elemental{phi}),
        make_global_parameter(elemental{psi}),
        const_builder.dimensions(d1(), d2(), d3(), nbQuadPt).initializer(jac).build(),
        const_builder.dimensions(d1(), d2(), d3(), b1, b2, b3).value(f).build(),
        result);
    verify([=](int i, int j, int k, int, int, int) { return (jac(i, j, k, 0) + jac(i, j, k, 1)) * phi * psi * f; },
        result);
}
