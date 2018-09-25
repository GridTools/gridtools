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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/meta.hpp"

namespace gridtools {

    /**
     * Class to specify access extents for stencil functions
     */
    template <int_t IMinus = 0,
        int_t IPlus = 0,
        int_t JMinus = 0,
        int_t JPlus = 0,
        int_t KMinus = 0,
        int_t KPlus = 0,
        int_t...>
    struct extent {
        using type = extent;

        typedef static_int<IMinus> iminus;
        typedef static_int<IPlus> iplus;
        typedef static_int<JMinus> jminus;
        typedef static_int<JPlus> jplus;
        typedef static_int<KMinus> kminus;
        typedef static_int<KPlus> kplus;
    };

    /**
     * Metafunction to check if a type is a extent
     */
    template <typename>
    struct is_extent : std::false_type {};

    template <int_t... Is>
    struct is_extent<extent<Is...>> : std::true_type {};

    template <typename T>
    struct is_extent<const T> : is_extent<T> {};

    /**
     * Metafunction taking extents and yielding an extent containing them
     */
    GT_META_LAZY_NAMESPASE {
        template <class...>
        struct enclosing_extent;
    }
    GT_META_DELEGATE_TO_LAZY(enclosing_extent, class... Extents, Extents...);
    GT_META_LAZY_NAMESPASE {
        template <>
        struct enclosing_extent<> {
            using type = extent<>;
        };

        template <int_t... Is>
        struct enclosing_extent<extent<Is...>> {
            using type = extent<Is...>;
        };

        template <int_t L_I_Minus,
            int_t L_I_Plus,
            int_t L_J_Minus,
            int_t L_J_Plus,
            int_t L_K_Minus,
            int_t L_K_Plus,
            int_t R_I_Minus,
            int_t R_I_Plus,
            int_t R_J_Minus,
            int_t R_J_Plus,
            int_t R_K_Minus,
            int_t R_K_Plus>
        struct enclosing_extent<extent<L_I_Minus, L_I_Plus, L_J_Minus, L_J_Plus, L_K_Minus, L_K_Plus>,
            extent<R_I_Minus, R_I_Plus, R_J_Minus, R_J_Plus, R_K_Minus, R_K_Plus>> {
            using type = extent<(L_I_Minus < R_I_Minus) ? L_I_Minus : R_I_Minus,
                (L_I_Plus < R_I_Plus) ? R_I_Plus : L_I_Plus,
                (L_J_Minus < R_J_Minus) ? L_J_Minus : R_J_Minus,
                (L_J_Plus < R_J_Plus) ? R_J_Plus : L_J_Plus,
                (L_K_Minus < R_K_Minus) ? L_K_Minus : R_K_Minus,
                (L_K_Plus < R_K_Plus) ? R_K_Plus : L_K_Plus>;
        };

        template <class... Extents>
        struct enclosing_extent : meta::lazy::combine<gridtools::enclosing_extent, meta::list<Extents...>> {};
    }

    /**
     *  Version of enclosing_extent with exactly two parameters
     *  It can be used to pass to MPL algorithms.
     */
    template <class Lhs, class Rhs>
    struct enclosing_extent_2 : lazy::enclosing_extent<Lhs, Rhs> {};

    /**
     * Metafunction taking two extents and yielding a extent which is the extension of one another
     */
    template <typename Extent1, typename Extent2>
    struct sum_extent {
        GRIDTOOLS_STATIC_ASSERT(is_extent<Extent1>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_extent<Extent2>::value, GT_INTERNAL_ERROR);

        using type = extent<Extent1::iminus::value + Extent2::iminus::value,
            Extent1::iplus::value + Extent2::iplus::value,
            Extent1::jminus::value + Extent2::jminus::value,
            Extent1::jplus::value + Extent2::jplus::value,
            Extent1::kminus::value + Extent2::kminus::value,
            Extent1::kplus::value + Extent2::kplus::value>;
    };
} // namespace gridtools
