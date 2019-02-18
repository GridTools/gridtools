/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/meta/type_traits.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;
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
  iteration point moves, so their values are constant. This is a typical example of global_parameter/
  global_accessor use.
*/

struct integration {
    using phi_t = global_accessor<0>;
    using psi_t = global_accessor<1>;
    using jac = in_accessor<2, extent<>, 4>;
    using f = in_accessor<3, extent<>, 6>;
    using result = inout_accessor<4, extent<>, 6>;

    using param_list = make_param_list<phi_t, psi_t, jac, f, result>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        dimension<1> i;
        dimension<2> j;
        dimension<3> k;
        dimension<4> di;
        dimension<5> dj;
        dimension<6> dk;
        dimension<4> qp;
        phi_t phi;
        psi_t psi;
        // projection of f on a (e.g.) P1 FE space:
        // loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
        // computational complexity in the order of  {(I) x (J) x (K) x (i) x (j) x (k) x (nq)}
        for (short_t I = 0; I < 2; ++I)
            for (short_t J = 0; J < 2; ++J)
                for (short_t K = 0; K < 2; ++K) {
                    for (short_t q = 0; q < 2; ++q) {
                        eval(result(di + I, dj + J, dk + K)) +=
                            eval(phi(I, J, K, q) * psi(0, 0, 0, q) * jac{i, j, k, qp + q} * f{i, j, k, di, dj, dk} +
                                 phi(I, J, K, q) * psi(0, 0, 0, q) * jac{i, j, k, qp + q} * f{i, j, k, di + 1, dj, dk} +
                                 phi(I, J, K, q) * psi(1, 0, 0, q) * jac{i, j, k, qp + q} * f{i, j, k, di, dj + 1, dk} +
                                 phi(I, J, K, q) * psi(1, 0, 0, q) * jac{i, j, k, qp + q} * f{i, j, k, di, dj, dk + 1} +
                                 phi(I, J, K, q) * psi(1, 1, 0, q) * jac{i, j, k, qp + q} *
                                     f{i, j, k, di + 1, dj + 1, dk} +
                                 phi(I, J, K, q) * psi(1, 0, 1, q) * jac{i, j, k, qp + q} *
                                     f{i, j, k, di + 1, dj, dk + 1} +
                                 phi(I, J, K, q) * psi(0, 1, 1, q) * jac{i, j, k, qp + q} *
                                     f{i, j, k, di, dj + 1, dk + 1} +
                                 phi(I, J, K, q) * psi(1, 1, 1, q) * jac{i, j, k, qp + q} *
                                     f{i, j, k, di + 1, dj + 1, dk + 1}) /
                            8;
                    }
                }
    }
};

struct extended_4d : regression_fixture<> {
    using layout_map_t = conditional_t<std::is_same<backend_t::backend_id_t, target::x86>::value,
        layout_map<3, 4, 5, 0, 1, 2>,
        layout_map<5, 4, 3, 2, 1, 0>>;
    using layout_map_quad_t = conditional_t<std::is_same<backend_t::backend_id_t, target::x86>::value,
        layout_map<1, 2, 3, 0>,
        layout_map<3, 2, 1, 0>>;

    template <unsigned Id, typename Layout>
    using special_storage_info_t = storage_tr::custom_layout_storage_info_t<Id, Layout>;

    using storage_t = storage_tr::data_store_t<float_type, special_storage_info_t<0, layout_map_t>>;
    using storage_global_quad_t = storage_tr::data_store_t<float_type, special_storage_info_t<1, layout_map_quad_t>>;

    static constexpr uint_t nbQuadPt = 2;
    static constexpr uint_t b1 = 2;
    static constexpr uint_t b2 = 2;
    static constexpr uint_t b3 = 2;

    template <class T = float_type>
    storage_t make_storage(T &&obj = {}) {
        return {{d1(), d2(), d3(), b1, b2, b3}, std::forward<T>(obj)};
    }

    /**
     * this is a user-defined class which will be used from within the user functor
     * by calling its  operator(). It can represent in this case values which are local to the elements
     * e.g. values of the basis functions in the quad points.
     */
    struct elemental {
        GT_FUNCTION double operator()(int, int, int, int) const { return m_val; }

        double m_val;
    };
};

TEST_F(extended_4d, test) {
    using global_par_storage_t = decltype(backend_t::make_global_parameter(elemental{}));
    arg<0, global_par_storage_t> p_phi;
    arg<1, global_par_storage_t> p_psi;
    arg<2, storage_global_quad_t> p_jac;
    arg<3, storage_t> p_f;
    arg<4, storage_t> p_result;

    float_type phi = 10, psi = 11, f = 1.3;
    auto jac = [](int i, int j, int k, int q) { return 1. + q; };
    auto ref = [=](int i, int j, int k, int I, int J, int K) {
        float_type res = 0;
        for (int q = 0; q < 2; ++q)
            res += (phi * psi * jac(i, j, k, q) * f + phi * psi * jac(i, j, k, q) * f +
                       phi * psi * jac(i, j, k, q) * f + phi * psi * jac(i, j, k, q) * f +
                       phi * psi * jac(i, j, k, q) * f + phi * psi * jac(i, j, k, q) * f +
                       phi * psi * jac(i, j, k, q) * f + phi * psi * jac(i, j, k, q) * f) /
                   8;
        return res;
    };
    auto result = make_storage();

    make_computation(p_phi = backend_t::make_global_parameter(elemental{phi}),
        p_psi = backend_t::make_global_parameter(elemental{psi}),
        p_jac = storage_global_quad_t{{d1(), d2(), d3(), nbQuadPt}, jac},
        p_f = make_storage(f),
        p_result = result,
        make_multistage(execute::forward(), make_stage<integration>(p_phi, p_psi, p_jac, p_f, p_result)))
        .run();

    verify(make_storage(ref), result);
}
