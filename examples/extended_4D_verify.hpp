/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include <tools/verifier.hpp>
#include <common/defs.hpp>

using namespace gridtools;

template < typename StorageLocal, typename StorageGlobal, typename Storage, typename Grid >
bool do_verification(uint_t d1, uint_t d2, uint_t d3, Storage const &result_, Grid const &grid) {

    typedef Storage storage_t;
    typedef StorageLocal storage_local_quad_t;
    typedef StorageGlobal storage_global_quad_t;

    uint_t nbQuadPt = 2; // referenceFE_Type::nbQuadPt;
    uint_t b1 = 2;
    uint_t b2 = 2;
    uint_t b3 = 2;

    typename storage_local_quad_t::storage_info_type meta_local_(b1, b2, b3, nbQuadPt);
    storage_local_quad_t phi(meta_local_, 0., "phi");
    storage_local_quad_t psi(meta_local_, 0., "psi");

    // I might want to treat it as a temporary storage (will use less memory but constantly copying back and forth)
    // Or alternatively computing the values on the quadrature points on the GPU
    typename storage_global_quad_t::storage_info_type meta_global_(d1, d2, d3, nbQuadPt);
    storage_global_quad_t jac(meta_global_, 0., "jac");

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

    typename storage_t::storage_info_type meta_(d1, d2, d3, b1, b2, b3);
    storage_t f(meta_, (float_type)1.3, "f");

    storage_t reference(meta_, (float_type)0., "result");

    for (int_t i = 1; i < d1 - 2; ++i)
        for (int_t j = 1; j < d2 - 2; ++j)
            for (int_t k = 0; k < d3 - 1; ++k)
                for (short_t I = 0; I < 2; ++I)
                    for (short_t J = 0; J < 2; ++J)
                        for (short_t K = 0; K < 2; ++K) {
                            // check the initialization to 0
                            assert(reference(i, j, k, I, J, K) == 0.);
                            for (short_t q = 0; q < 2; ++q) {
                                reference(i, j, k, I, J, K) +=
                                    (phi(I, J, K, q) * psi(0, 0, 0, q) * jac(i, j, k, q) * f(i, j, k, 0, 0, 0) +
                                        phi(I, J, K, q) * psi(1, 0, 0, q) * jac(i, j, k, q) * f(i, j, k, 1, 0, 0) +
                                        phi(I, J, K, q) * psi(0, 1, 0, q) * jac(i, j, k, q) * f(i, j, k, 0, 1, 0) +
                                        phi(I, J, K, q) * psi(0, 0, 1, q) * jac(i, j, k, q) * f(i, j, k, 0, 0, 1) +
                                        phi(I, J, K, q) * psi(1, 1, 0, q) * jac(i, j, k, q) * f(i, j, k, 1, 1, 0) +
                                        phi(I, J, K, q) * psi(1, 1, 0, q) * jac(i, j, k, q) * f(i, j, k, 1, 0, 1) +
                                        phi(I, J, K, q) * psi(0, 1, 1, q) * jac(i, j, k, q) * f(i, j, k, 0, 1, 1) +
                                        phi(I, J, K, q) * psi(1, 1, 1, q) * jac(i, j, k, q) * f(i, j, k, 1, 1, 1)) /
                                    8;
                            }
                        }

#ifdef CXX11_ENABLED
#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array< array< uint_t, 2 >, 6 > halos{{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}};
    bool result = verif.verify(grid, reference, result_, halos);
#else
#if FLOAT_PRECISION == 4
    verifier verif(1e-6, 0);
#else
    verifier verif(1e-12, 0);
#endif
    bool result = verif.verify(grid, reference, result_);
#endif
    return result;
}
