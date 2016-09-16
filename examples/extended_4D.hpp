/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <stencil-composition/stencil-composition.hpp>
#include "Options.hpp"
#include "extended_4D_verify.hpp"

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

//                      dims  x y z  qp
//                   strides  1 x xy xyz
typedef gridtools::layout_map< 3, 2, 1, 0 > layout4_t;
typedef gridtools::layout_map< 2, 1, 0, 3, 4, 5 > layout_t;

typedef BACKEND::storage_info< __COUNTER__, layout_t > metadata_t;
typedef BACKEND::storage_info< __COUNTER__, layout4_t > metadata_global_quad_t;
typedef BACKEND::storage_info< __COUNTER__, layout4_t > metadata_local_quad_t;
typedef BACKEND::storage_type< float_type, metadata_t >::type storage_type;
typedef BACKEND::storage_type< float_type, metadata_global_quad_t >::type storage_global_quad_t;
typedef BACKEND::storage_type< float_type, metadata_local_quad_t >::type storage_local_quad_t;

/**
  @file
  @brief This file shows a possible usage of the extension to storages with more than 3 space dimensions.

  We recall that the space dimensions simply identify the number of indexes/strides required to access
  a contiguous chunck of storage. The number of space dimensions is fully arbitrary.

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

  In this example we introduce also another syntactic element in the high level expression: the operator exclamation
  mark (!). This operator prefixed to a placeholder means that the corresponding storage index is not considered, and
  only the offsets are used to get the absolute address. This allows to perform operations which are not stencil-like.
  It is used in this case to address the basis functions values.
*/

namespace assembly {

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct integration {
        typedef in_accessor< 0, extent<>, 4 > phi;
        typedef in_accessor< 1, extent<>, 4 > psi; // how to detect when index is wrong??
        typedef in_accessor< 2, extent<>, 4 > jac;
        typedef in_accessor< 3, extent<>, 6 > f;
        typedef inout_accessor< 4, extent<>, 6 > result;
        typedef boost::mpl::vector< phi, psi, jac, f, result > arg_list;
        using quad = dimension< 4 >;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            x::Index i;
            y::Index j;
            z::Index k;
            dimension< 4 >::Index di;
            dimension< 5 >::Index dj;
            dimension< 6 >::Index dk;
            quad::Index qp;
            // projection of f on a (e.g.) P1 FE space:
            // loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
            // computational complexity in the order of  {(I) x (J) x (K) x (i) x (j) x (k) x (nq)}
            for (short_t I = 0; I < 2; ++I)
                for (short_t J = 0; J < 2; ++J)
                    for (short_t K = 0; K < 2; ++K) {
                        // check the initialization to 0
                        assert(eval(result{di + I, dj + J, dk + K}) == 0.);
                        for (short_t q = 0; q < 2; ++q) {
                            eval(result{di + I, dj + J, dk + K}) +=
                                eval(!phi{i + I, j + J, k + K, qp + q} * !psi{qp + q} * jac{qp + q} * f{} +
                                     !phi{i + I, j + J, k + K, qp + q} * !psi{i + 1, qp + q} * jac{qp + q} * f{di + 1} +
                                     !phi{i + I, j + J, k + K, qp + q} * !psi{j + 1, qp + q} * jac{qp + q} * f{dj + 1} +
                                     !phi{i + I, j + J, k + K, qp + q} * !psi{k + 1, qp + q} * jac{qp + q} * f{dk + 1} +
                                     !phi{i + I, j + J, k + K, qp + q} * !psi{i + 1, j + 1, qp + q} * jac{qp + q} *
                                         f{di + 1, dj + 1} +
                                     !phi{i + I, j + J, k + K, qp + q} * !psi{i + 1, k + 1, qp + q} * jac{qp + q} *
                                         f{di + 1, dk + 1} +
                                     !phi{i + I, j + J, k + K, qp + q} * !psi{j + 1, k + 1, qp + q} * jac{qp + q} *
                                         f{dj + 1, dk + 1} +
                                     !phi{i + I, j + J, k + K, qp + q} * !psi{i + 1, j + 1, k + 1, qp + q} *
                                         jac{qp + q} * f{di + 1, dj + 1, dk + 1}) /
                                8;
                        }
                    }
        }
    };

    std::ostream &operator<<(std::ostream &s, integration const) { return s << "integration"; }

    bool test(uint_t d1, uint_t d2, uint_t d3) {

        typedef arg< 0, storage_local_quad_t > p_phi;
        typedef arg< 1, storage_local_quad_t > p_psi;
        typedef arg< 2, storage_global_quad_t > p_jac;
        typedef arg< 3, storage_type > p_f;
        typedef arg< 4, storage_type > p_result;

        typedef boost::mpl::vector< p_phi, p_psi, p_jac, p_f, p_result > accessor_list;

        uint_t nbQuadPt = 2; // referenceFE_Type::nbQuadPt;
        uint_t b1 = 2;
        uint_t b2 = 2;
        uint_t b3 = 2;
        // basis functions available in a 2x2x2 cell, because of P1 FE
        metadata_local_quad_t local_metadata(b1, b2, b3, nbQuadPt);

        storage_local_quad_t phi(local_metadata, 0., "phi");
        storage_local_quad_t psi(local_metadata, 0., "psi");

        // I might want to treat it as a temporary storage (will use less memory but constantly copying back and forth)
        // Or alternatively computing the values on the quadrature points on the GPU
        metadata_global_quad_t integration_metadata(d1, d2, d3, nbQuadPt);
        storage_global_quad_t jac(integration_metadata, 0., "jac");

        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k)
                    for (uint_t q = 0; q < nbQuadPt; ++q) {
                        jac(i, j, k, q) = 1. + q;
                    }
        for (uint_t i = 0; i < b1; ++i)
            for (uint_t j = 0; j < b2; ++j)
                for (uint_t k = 0; k < b3; ++k)
                    for (uint_t q = 0; q < nbQuadPt; ++q) {
                        phi(i, j, k, q) = 10.;
                        psi(i, j, k, q) = 11.;
                    }

        metadata_t meta_(d1, d2, d3, b1, b2, b3);
        storage_type f(meta_, (float_type)1.3, "f");
        storage_type result(meta_, (float_type)0., "result");

        gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&phi, &psi, &jac, &f, &result));
        /**
           - Definition of the physical dimensions of the problem.
           The grid constructor takes the horizontal plane dimensions,
           hile the vertical ones are set according the the axis property soon after
        */
        uint_t di[5] = {1, 1, 1, d1 - 3, d1};
        uint_t dj[5] = {1, 1, 1, d2 - 3, d2};
        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 2;

#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        stencil *
#else
        boost::shared_ptr< gridtools::stencil >
#endif
#endif
            fe_comp = make_computation< gridtools::BACKEND >(
                domain,
                grid,
                make_multistage               //! \todo all the arguments in the call to make_mss are actually dummy.
                (execute< forward >(), //!\todo parameter used only for overloading purpose?
                    make_stage< integration >(p_phi(), p_psi(), p_jac(), p_f(), p_result())));

        fe_comp->ready();
        fe_comp->steady();
        fe_comp->run();
        fe_comp->finalize();

        return do_verification< storage_local_quad_t, storage_global_quad_t >(d1, d2, d3, result, grid);
    }

}; // namespace extended_4d
