/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * loop_interval.h
 *
 *  Created on: Feb 17, 2015
 *      Author: carlosos
 */

#pragma once

#include <tuple>

#include "../meta/filter.hpp"
#include "../meta/first.hpp"
#include "../meta/id.hpp"
#include "../meta/length.hpp"
#include "../meta/logical.hpp"
#include "../meta/macros.hpp"
#include "../meta/type_traits.hpp"
#include "./caches/cache_traits.hpp"
#include "./esf.hpp"

namespace gridtools {
    namespace _impl {
        template <class T, class = void>
        struct is_sequence_of_caches : std::false_type {};

        template <template <class...> class L, class... Ts>
        struct is_sequence_of_caches<L<Ts...>> : conjunction<is_cache<Ts>...> {};
    } // namespace _impl

    /**
     * @struct is_mss_parameter
     * metafunction that determines if a given type is a valid parameter for mss_descriptor
     */
    template <class T>
    using is_mss_parameter = bool_constant<_impl::is_sequence_of_caches<T>::value || is_esf_descriptor<T>::value>;

    /**
     * @struct extract_mss_caches
     * metafunction that extracts from a sequence of mss descriptor parameters, a sequence of all caches
     */
    template <typename... MssParameters>
    struct extract_mss_caches {
#ifdef GT_DISABLE_CACHING
        typedef std::tuple<> type;
#else
        using tuple_of_caches = meta::filter<_impl::is_sequence_of_caches, std::tuple<MssParameters...>>;

        GT_STATIC_ASSERT(meta::length<tuple_of_caches>::value < 2,
            "Wrong number of sequence of caches. Probably caches are defined in multiple dinstinct instances of "
            "define_caches\n"
            "Only one instance of define_caches is allowed.");

        using type = typename conditional_t<meta::length<tuple_of_caches>::value == 0,
            meta::lazy::id<std::tuple<>>,
            meta::lazy::first<tuple_of_caches>>::type;
#endif
    };

    /**
     * @struct extract_mss_esfs
     * metafunction that extracts from a sequence of mss descriptor parameters, a sequence of all esf descriptors
     */
    template <class... Ts>
    using extract_mss_esfs = meta::filter<is_esf_descriptor, std::tuple<Ts...>>;

} // namespace gridtools
