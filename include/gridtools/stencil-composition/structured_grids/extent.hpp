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

#include <boost/mpl/assert.hpp>
#include <boost/mpl/min_max.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/mpl/reverse_fold.hpp>
#include <boost/mpl/vector.hpp>

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

    template <typename In>
    struct is_staggered : public boost::false_type {};

    template <int_t... Grid>
    struct staggered : public extent<Grid...> {};

    template <int_t... Grid>
    struct is_staggered<staggered<Grid...>> : public boost::true_type {};

    /**
     * Metafunction to check if a type is a extent
     */
    template <typename T>
    struct is_extent : boost::false_type {};

    /**
     * Metafunction to check if a type is a extent - Specialization yielding true
     */
    template <int_t... Is>
    struct is_extent<extent<Is...>> : boost::true_type {};

    /**
     * Metafunction to check if a type is a extent - Specialization yielding true
     */
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
            int_t... Ls,
            int_t R_I_Minus,
            int_t R_I_Plus,
            int_t R_J_Minus,
            int_t R_J_Plus,
            int_t R_K_Minus,
            int_t R_K_Plus,
            int_t... Rs>
        struct enclosing_extent<extent<L_I_Minus, L_I_Plus, L_J_Minus, L_J_Plus, L_K_Minus, L_K_Plus, Ls...>,
            extent<R_I_Minus, R_I_Plus, R_J_Minus, R_J_Plus, R_K_Minus, R_K_Plus, Rs...>> {
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
     *  Verison of enclosing_extent with exactly two parameters
     */
    template <class Lhs, class Rhs>
    struct enclosing_extent_2 : lazy::enclosing_extent<Lhs, Rhs> {};

    /**
     * Metafunction taking two extents and yielding a extent which is the extension of one another
     */
    template <typename Extent1, typename Extent2>
    struct sum_extent {
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::or_<is_extent<Extent1>, is_staggered<Extent1>>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::or_<is_extent<Extent2>, is_staggered<Extent2>>::value), GT_INTERNAL_ERROR);

        typedef extent<boost::mpl::plus<typename Extent1::iminus, typename Extent2::iminus>::type::value,
            boost::mpl::plus<typename Extent1::iplus, typename Extent2::iplus>::type::value,
            boost::mpl::plus<typename Extent1::jminus, typename Extent2::jminus>::type::value,
            boost::mpl::plus<typename Extent1::jplus, typename Extent2::jplus>::type::value,
            boost::mpl::plus<typename Extent1::kminus, typename Extent2::kminus>::type::value,
            boost::mpl::plus<typename Extent1::kplus, typename Extent2::kplus>::type::value>
            type;
    };

    template <typename Extent>
    struct extent_get_iminus {
        GRIDTOOLS_STATIC_ASSERT((is_extent<Extent>::value), GT_INTERNAL_ERROR);
        static const int_t value = Extent::iminus::value;
    };
    template <typename Extent>
    struct extent_get_iplus {
        GRIDTOOLS_STATIC_ASSERT((is_extent<Extent>::value), GT_INTERNAL_ERROR);
        static const int_t value = Extent::iplus::value;
    };
    template <typename Extent>
    struct extent_get_jminus {
        GRIDTOOLS_STATIC_ASSERT((is_extent<Extent>::value), GT_INTERNAL_ERROR);
        static const int_t value = Extent::jminus::value;
    };
    template <typename Extent>
    struct extent_get_jplus {
        GRIDTOOLS_STATIC_ASSERT((is_extent<Extent>::value), GT_INTERNAL_ERROR);
        static const int_t value = Extent::jplus::value;
    };

} // namespace gridtools
