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

#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

#ifdef __CUDACC__
#define BACKEND backend< enumtype::Cuda, enumtype::GRIDBACKEND, enumtype::Block >
template < unsigned Id, typename Layout >
using special_metadata_t = gridtools::cuda_storage_info< Id, Layout >;
typedef special_metadata_t< 0, gridtools::layout_map< 5, 4, 3, 2, 1, 0 > > metadata_t;
typedef special_metadata_t< 1, gridtools::layout_map< 3, 2, 1, 0 > > metadata_global_quad_t;
typedef special_metadata_t< 2, gridtools::layout_map< -1, -1, -1, 3, 2, 1, 0 > > metadata_local_quad_t;
#else
#define BACKEND backend< enumtype::Host, enumtype::GRIDBACKEND, enumtype::Block >
template < unsigned Id, typename Layout >
using special_metadata_t = gridtools::host_storage_info< Id, Layout >;
typedef special_metadata_t< 0, gridtools::layout_map< 3, 4, 5, 0, 1, 2 > > metadata_t;
typedef special_metadata_t< 1, gridtools::layout_map< 1, 2, 3, 0 > > metadata_global_quad_t;
typedef special_metadata_t< 2, gridtools::layout_map< -1, -1, -1, 1, 2, 3, 0 > > metadata_local_quad_t;
#endif

//                      dims  x y z  qp
//                   strides  1 x xy xyz
typedef layout_map< -1, -1, -1, 3, 2, 1, 0 > layoutphi_t;
typedef layout_map< 3, 2, 1, 0 > layout4_t;
typedef layout_map< 2, 1, 0, 3, 4, 5 > layout_t;

typedef BACKEND::storage_traits_t::data_store_t< float_type, metadata_t > storage_type;
typedef BACKEND::storage_traits_t::data_store_t< float_type, metadata_global_quad_t > storage_global_quad_t;
typedef BACKEND::storage_traits_t::data_store_t< float_type, metadata_local_quad_t > storage_local_quad_t;

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

  This code uses a deprecated way of dealing with input data that does not 'move' with the iteration space.
  The right way of doing it would be to use global_accessors and global_parameters (the proper implementation
  can be found in the example folder). Here Phi and Psi are then 7 dimensional arrays, where the first three
  dimensions are 'killed' (the corresponding layout map entries are set to -1). In this way, when the iteration
  point is moved the fields phi and psi offset if not updated and then the other 4 dimensions are accessed
  from within the stencil operator. The values accessed there are always the same regardless of the iteration
  point.
*/

template < typename StorageLocal, typename StorageGlobal, typename Storage, typename Grid >
bool do_verification(uint_t d1, uint_t d2, uint_t d3, Storage const &result_, Grid const &grid) {

    typedef Storage storage_t;
    typedef StorageLocal storage_local_quad_t;
    typedef StorageGlobal storage_global_quad_t;

    uint_t nbQuadPt = 2; // referenceFE_Type::nbQuadPt;
    uint_t b1 = 2;
    uint_t b2 = 2;
    uint_t b3 = 2;

    metadata_local_quad_t local_metadata(1, 1, 1, b1, b2, b3, nbQuadPt);

    storage_local_quad_t phi(local_metadata, 0., "phi");
    storage_local_quad_t psi(local_metadata, 0., "psi");

    // I might want to treat it as a temporary storage (will use less memory but constantly copying back and forth)
    // Or alternatively computing the values on the quadrature points on the GPU
    metadata_global_quad_t integration_metadata(d1, d2, d3, nbQuadPt);
    storage_global_quad_t jac(integration_metadata, 0., "jac");

    auto jacv = make_host_view(jac);
    auto phiv = make_host_view(phi);
    auto psiv = make_host_view(psi);

    for (uint_t i = 0; i < d1; ++i)
        for (uint_t j = 0; j < d2; ++j)
            for (uint_t k = 0; k < d3; ++k)
                for (uint_t q = 0; q < nbQuadPt; ++q) {
                    jacv(i, j, k, q) = 1. + q;
                }
    for (uint_t i = 0; i < b1; ++i)
        for (uint_t j = 0; j < b2; ++j)
            for (uint_t k = 0; k < b3; ++k)
                for (uint_t q = 0; q < nbQuadPt; ++q) {
                    phiv(0, 0, 0, i, j, k, q) = 10.;
                    psiv(0, 0, 0, i, j, k, q) = 11.;
                }

    metadata_t meta_(d1, d2, d3, b1, b2, b3);
    storage_t f(meta_, (float_type)1.3, "f");
    auto fv = make_host_view(f);

    storage_t reference(meta_, (float_type)0., "result");
    auto referencev = make_host_view(reference);

    for (int_t i = 1; i < d1 - 2; ++i)
        for (int_t j = 1; j < d2 - 2; ++j)
            for (int_t k = 0; k < d3 - 1; ++k)
                for (short_t I = 0; I < 2; ++I)
                    for (short_t J = 0; J < 2; ++J)
                        for (short_t K = 0; K < 2; ++K) {
                            // check the initialization to 0
                            assert(referencev(i, j, k, I, J, K) == 0.);
                            for (short_t q = 0; q < 2; ++q) {
                                referencev(i, j, k, I, J, K) +=
                                    (phiv(1, 1, 1, I, J, K, q) * psiv(1, 1, 1, 0, 0, 0, q) * jacv(i, j, k, q) *
                                            fv(i, j, k, 0, 0, 0) +
                                        phiv(1, 1, 1, I, J, K, q) * psiv(1, 1, 1, 1, 0, 0, q) * jacv(i, j, k, q) *
                                            fv(i, j, k, 1, 0, 0) +
                                        phiv(1, 1, 1, I, J, K, q) * psiv(1, 1, 1, 0, 1, 0, q) * jacv(i, j, k, q) *
                                            fv(i, j, k, 0, 1, 0) +
                                        phiv(1, 1, 1, I, J, K, q) * psiv(1, 1, 1, 0, 0, 1, q) * jacv(i, j, k, q) *
                                            fv(i, j, k, 0, 0, 1) +
                                        phiv(1, 1, 1, I, J, K, q) * psiv(1, 1, 1, 1, 1, 0, q) * jacv(i, j, k, q) *
                                            fv(i, j, k, 1, 1, 0) +
                                        phiv(1, 1, 1, I, J, K, q) * psiv(1, 1, 1, 1, 1, 0, q) * jacv(i, j, k, q) *
                                            fv(i, j, k, 1, 0, 1) +
                                        phiv(1, 1, 1, I, J, K, q) * psiv(1, 1, 1, 0, 1, 1, q) * jacv(i, j, k, q) *
                                            fv(i, j, k, 0, 1, 1) +
                                        phiv(1, 1, 1, I, J, K, q) * psiv(1, 1, 1, 1, 1, 1, q) * jacv(i, j, k, q) *
                                            fv(i, j, k, 1, 1, 1)) /
                                    8;
                            }
                        }

#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array< array< uint_t, 2 >, 6 > halos{{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}};
    bool result = verif.verify(grid, reference, result_, halos);

    return result;
}

namespace assembly {

    typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct integration {
        typedef in_accessor< 0, extent<>, 7 > phi;
        typedef in_accessor< 1, extent<>, 7 > psi; // how to detect when index is wrong??
        typedef in_accessor< 2, extent<>, 4 > jac;
        typedef in_accessor< 3, extent<>, 6 > f;
        typedef inout_accessor< 4, extent<>, 6 > result;
        typedef boost::mpl::vector< phi, psi, jac, f, result > arg_list;
        using quad = dimension< 7 >;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            dimension< 1 > i;
            dimension< 2 > j;
            dimension< 3 > k;
            dimension< 4 > di;
            dimension< 5 > dj;
            dimension< 6 > dk;
            quad qp;
            // projection of f on a (e.g.) P1 FE space:
            // loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
            // computational complexity in the order of  {(I) x (J) x (K) x (i) x (j) x (k) x (nq)}
            for (short_t I = 0; I < 2; ++I)
                for (short_t J = 0; J < 2; ++J)
                    for (short_t K = 0; K < 2; ++K) {
                        // check the initialization to 0
                        assert(eval(result{i, j, k, di + I, dj + J, dk + K}) == 0.);
                        for (short_t q = 0; q < 2; ++q) {
                            eval(result{di + I, dj + J, dk + K}) +=
                                eval(phi{i + I, j + J, k + K, qp + q} * psi{i, j, k, qp + q} * jac{i, j, k, di + q} *
                                         f{i, j, k, di, dj, dk} +
                                     phi{di + I, dj + J, dk + K, qp + q} * psi{di + 1, dj, dk, qp + q} *
                                         jac{i, j, k, di + q} * f{i, j, k, di + 1, dj, dk} +
                                     phi{di + I, dj + J, dk + K, qp + q} * psi{dj + 1, dj, dk, qp + q} *
                                         jac{i, j, k, di + q} * f{i, j, k, di, dj + 1, dk} +
                                     phi{di + I, dj + J, dk + K, qp + q} * psi{dk + 1, dj, dk, qp + q} *
                                         jac{i, j, k, di + q} * f{i, k, k, di, dj, dk + 1} +
                                     phi{di + I, dj + J, dk + K, qp + q} * psi{di + 1, dj + 1, dk, qp + q} *
                                         jac{i, j, k, di + q} * f{i, j, k, di + 1, dj + 1, dk} +
                                     phi{di + I, dj + J, dk + K, qp + q} * psi{di + 1, dj, dk + 1, qp + q} *
                                         jac{i, j, k, di + q} * f{i, j, k, di + 1, dj, dk + 1} +
                                     phi{di + I, dj + J, dk + K, qp + q} * psi{di, dj + 1, dk + 1, qp + q} *
                                         jac{i, j, k, di + q} * f{i, j, k, di, dj + 1, dk + 1} +
                                     phi{di + I, dj + J, dk + K, qp + q} * psi{di + 1, dj + 1, dk + 1, qp + q} *
                                         jac{i, j, k, di + q} * f{i, j, k, di + 1, dj + 1, dk + 1}) /
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
        metadata_local_quad_t local_metadata(1, 1, 1, b1, b2, b3, nbQuadPt);

        storage_local_quad_t phi(local_metadata, 0., "phi");
        storage_local_quad_t psi(local_metadata, 0., "psi");

        // I might want to treat it as a temporary storage (will use less memory but constantly copying back and forth)
        // Or alternatively computing the values on the quadrature points on the GPU
        metadata_global_quad_t integration_metadata(d1, d2, d3, nbQuadPt);
        storage_global_quad_t jac(integration_metadata, 0., "jac");

        auto jacv = make_host_view(jac);
        auto phiv = make_host_view(phi);
        auto psiv = make_host_view(psi);

        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k)
                    for (uint_t q = 0; q < nbQuadPt; ++q) {
                        jacv(i, j, k, q) = 1. + q;
                    }
        for (uint_t i = 0; i < b1; ++i)
            for (uint_t j = 0; j < b2; ++j)
                for (uint_t k = 0; k < b3; ++k)
                    for (uint_t q = 0; q < nbQuadPt; ++q) {
                        phiv(0, 0, 0, i, j, k, q) = 10.;
                        psiv(0, 0, 0, i, j, k, q) = 11.;
                    }

        metadata_t meta_(d1, d2, d3, b1, b2, b3);
        storage_type f(meta_, (float_type)1.3, "f");
        storage_type result(meta_, (float_type)0., "result");

        aggregator_type< accessor_list > domain(phi, psi, jac, f, result);
        /**
           - Definition of the physical dimensions of the problem.
           The grid constructor takes the horizontal plane dimensions,
           hile the vertical ones are set according the the axis property soon after
        */
        uint_t di[5] = {1, 1, 1, d1 - 3, d1};
        uint_t dj[5] = {1, 1, 1, d2 - 3, d2};
        grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 2;

        auto fe_comp =
            make_computation< BACKEND >(domain,
                grid,
                make_multistage        //! \todo all the arguments in the call to make_mss are actually dummy.
                (execute< forward >(), //!\todo parameter used only for overloading purpose?
                                            make_stage< integration >(p_phi(), p_psi(), p_jac(), p_f(), p_result())));

        fe_comp->ready();
        fe_comp->steady();
        fe_comp->run();
        fe_comp->finalize();

        return do_verification< storage_local_quad_t, storage_global_quad_t >(d1, d2, d3, result, grid);
    }

}; // namespace extended_4d

TEST(Accessor, Multidimensional) { ASSERT_TRUE(assembly::test(13, 14, 12)); }
