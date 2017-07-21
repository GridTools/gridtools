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

    typename storage_local_quad_t::storage_info_t meta_local_(b1, b2, b3, nbQuadPt);
    storage_local_quad_t phi(meta_local_, 10.);
    storage_local_quad_t psi(meta_local_, 11.);

    // I might want to treat it as a temporary storage (will use less memory but constantly copying back and forth)
    // Or alternatively computing the values on the quadrature points on the GPU
    typename storage_global_quad_t::storage_info_t meta_global_(d1, d2, d3, nbQuadPt);
    storage_global_quad_t jac(meta_global_, [](int i, int j, int k, int q) { return 1. + q; });

    auto jacv = make_host_view(jac);
    auto phiv = make_host_view(phi);
    auto psiv = make_host_view(psi);

    typename storage_t::storage_info_t meta_(d1, d2, d3, b1, b2, b3);
    storage_t f(meta_, (float_type)1.3);
    auto fv = make_host_view(f);

    storage_t reference(meta_, (float_type)0.);
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
                                    (phiv(I, J, K, q) * psiv(0, 0, 0, q) * jacv(i, j, k, q) * fv(i, j, k, 0, 0, 0) +
                                        phiv(I, J, K, q) * psiv(1, 0, 0, q) * jacv(i, j, k, q) * fv(i, j, k, 1, 0, 0) +
                                        phiv(I, J, K, q) * psiv(0, 1, 0, q) * jacv(i, j, k, q) * fv(i, j, k, 0, 1, 0) +
                                        phiv(I, J, K, q) * psiv(0, 0, 1, q) * jacv(i, j, k, q) * fv(i, j, k, 0, 0, 1) +
                                        phiv(I, J, K, q) * psiv(1, 1, 0, q) * jacv(i, j, k, q) * fv(i, j, k, 1, 1, 0) +
                                        phiv(I, J, K, q) * psiv(1, 1, 0, q) * jacv(i, j, k, q) * fv(i, j, k, 1, 0, 1) +
                                        phiv(I, J, K, q) * psiv(0, 1, 1, q) * jacv(i, j, k, q) * fv(i, j, k, 0, 1, 1) +
                                        phiv(I, J, K, q) * psiv(1, 1, 1, q) * jacv(i, j, k, q) * fv(i, j, k, 1, 1, 1)) /
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
