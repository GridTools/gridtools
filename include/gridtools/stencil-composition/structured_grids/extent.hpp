/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../meta.hpp"

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
    GT_META_LAZY_NAMESPACE {
        template <class...>
        struct enclosing_extent;
    }
    GT_META_DELEGATE_TO_LAZY(enclosing_extent, class... Extents, Extents...);
    GT_META_LAZY_NAMESPACE {
        template <>
        struct enclosing_extent<> {
            using type = extent<>;
        };

        template <int_t... Is>
        struct enclosing_extent<extent<Is...>> {
            using type = extent<Is...>;
        };

        template <int_t IMinus1,
            int_t IPlus1,
            int_t JMinus1,
            int_t JPlus1,
            int_t KMinus1,
            int_t KPlus1,
            int_t IMinus2,
            int_t IPlus2,
            int_t JMinus2,
            int_t JPlus2,
            int_t KMinus2,
            int_t KPlus2>
        struct enclosing_extent<extent<IMinus1, IPlus1, JMinus1, JPlus1, KMinus1, KPlus1>,
            extent<IMinus2, IPlus2, JMinus2, JPlus2, KMinus2, KPlus2>> {
            using type = extent<(IMinus1 < IMinus2) ? IMinus1 : IMinus2,
                (IPlus1 < IPlus2) ? IPlus2 : IPlus1,
                (JMinus1 < JMinus2) ? JMinus1 : JMinus2,
                (JPlus1 < JPlus2) ? JPlus2 : JPlus1,
                (KMinus1 < KMinus2) ? KMinus1 : KMinus2,
                (KPlus1 < KPlus2) ? KPlus2 : KPlus1>;
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
        GT_STATIC_ASSERT(is_extent<Extent1>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_extent<Extent2>::value, GT_INTERNAL_ERROR);

        using type = extent<Extent1::iminus::value + Extent2::iminus::value,
            Extent1::iplus::value + Extent2::iplus::value,
            Extent1::jminus::value + Extent2::jminus::value,
            Extent1::jplus::value + Extent2::jplus::value,
            Extent1::kminus::value + Extent2::kminus::value,
            Extent1::kplus::value + Extent2::kplus::value>;
    };

    struct rt_extent {
        template <int_t IMinus, int_t IPlus, int_t JMinus, int_t JPlus, int_t KMinus, int_t KPlus, int_t... Rest>
        constexpr rt_extent(extent<IMinus, IPlus, JMinus, JPlus, KMinus, KPlus, Rest...>)
            : iminus(IMinus), iplus(IPlus), jminus(JMinus), jplus(JPlus), kminus(KMinus), kplus(KPlus) {}
        constexpr rt_extent(int_t iminus, int_t iplus, int_t jminus, int_t jplus, int_t kminus, int_t kplus)
            : iminus(iminus), iplus(iplus), jminus(jminus), jplus(jplus), kminus(kminus), kplus(kplus) {}
        rt_extent() = default;
        constexpr bool operator==(const rt_extent &rhs) const {
            return iminus == rhs.iminus && iplus == rhs.iplus && jminus == rhs.jminus && jplus == rhs.jplus &&
                   kminus == rhs.kminus && kplus == rhs.kplus;
        }
        int_t iminus = 0;
        int_t iplus = 0;
        int_t jminus = 0;
        int_t jplus = 0;
        int_t kminus = 0;
        int_t kplus = 0;
    };

} // namespace gridtools
