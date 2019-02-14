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

#include "defs.hpp"
#include <stdlib.h>

/** @file
@brief serie of loops unrolled at compile time
Recursive templates used to perform compile-time loop unrolling
*/

namespace gridtools {

    /** \ingroup common
        @{
        \defgroup ndloops n-Dimensional Loops
        @{
    */

    /** @brief Product of all the elements of a generic array accessed with the [] operator, whose dimension is a
     * compile-time constant*/
    template <int_t D>
    struct prod;

    template <>
    struct prod<-1> {
        template <typename ARRAY>
        int operator()(ARRAY const &dimensions) const {
            return 1;
        }
    };

    template <int_t D>
    struct prod {
        template <typename ARRAY>
        int operator()(ARRAY const &dimensions) const {
            //    std::cout << D << " prod    " << dimensions[D]*prod<D-1>()(dimensions) << "\n";
            return dimensions[D] * prod<D - 1>()(dimensions);
        }
    };

    /** @brief given two vectors \f$a\f$ and \f$b\f$ it implements: \f$\sum_i(a(i)\prod_{j=0}^{i-1}b(j))\f$ */
    template <int_t D>
    struct access_to;

    template <>
    struct access_to<1> {
        template <typename ARRAY>
        int operator()(ARRAY const &indices, ARRAY const &) const {
            // std:: cout << "indices[1] " << indices[1] << "\n";
            return indices[0];
        }
    };

    template <int_t D>
    struct access_to {
        template <typename ARRAY>
        int operator()(ARRAY const &indices, ARRAY const &dimensions) const {
            // std::cout << access_to<D-1>()(indices,dimensions) << " + "
            //          << indices[D-1] << " * "
            //          << prod<D-2>()(dimensions) << "\n";
            return access_to<D - 1>()(indices, dimensions) + indices[D - 1] * prod<D - 2>()(dimensions);
        }
    };

    struct bounds_sizes {
        uint_t imin;
        uint_t imax;
        uint_t sizes;
    };

    struct bounds {
        uint_t imin;
        uint_t imax;
    };

    /**@brief of each element of an array it performs a loop between the array bounds defined in a template parameter,
     * and it computes a function of type F */
    template <int_t I, typename F>
    struct access_loop;

    template <typename F>
    struct access_loop<0, F> {
        template <typename arraybounds, typename array>
        void operator()(arraybounds const &ab, array const &sizes, F &f, uint_t idx = 0) {
            f(idx);
        }
    };

    template <int_t I, typename F>
    struct access_loop {
        template <typename arraybounds, typename array>
        void operator()(arraybounds const &ab, array const &sizes, F &f, uint_t idx = 0) {
            uint_t midx;
            for (uint_t i = ab[I - 1].imin; i <= ab[I - 1].imax; ++i) {
                midx = idx + i * prod<I - 2>()(sizes);
                access_loop<I - 1, F>()(ab, sizes, f, midx);
            }
        }
    };

    template <int_t I>
    struct loop;

    template <>
    struct loop<0> {
        template <typename F, typename arraybounds, typename array>
        void operator()(arraybounds const &, F const &f, array &tuple) {
            f(tuple);
        }
    };

    template <int_t I>
    struct loop {
        template <typename F, typename arraybounds, typename array>
        void operator()(arraybounds const &ab, F const &f, array &tuple) {
            for (uint_t i = ab[I - 1].imin; i <= ab[I - 1].imax; ++i) {
                tuple[I - 1] = i;
                loop<I - 1>()(ab, f, tuple);
            }
        }
    };

    /** @brief similar to the previous struct, given the upper and lower bound */
    template <int_t I, int_t LB = -1, int_t UB = 1>
    struct neigh_loop;

    template <int_t LB, int_t UB>
    struct neigh_loop<0, LB, UB> {
        template <typename F, typename array>
        void operator()(F &f, array &tuple) {
            f(tuple);
        }
    };

    template <int_t I, int_t LB, int_t UB>
    struct neigh_loop {
        template <typename F, typename array>
        void operator()(F &f, array &tuple) {
            for (int i = LB; i <= UB; ++i) {
                tuple[I - 1] = i;
                neigh_loop<I - 1, LB, UB>()(f, tuple);
            }
        }
    };

    /** @} */
    /** @} */
} // namespace gridtools
