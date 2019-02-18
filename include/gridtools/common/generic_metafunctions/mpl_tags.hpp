/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

#include <boost/mpl/arithmetic.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/mpl/integral_c_tag.hpp>

namespace boost {
    namespace mpl {
        /** \ingroup common
            @{
            \ingroup allmeta
            @{
            \ingroup mplutil
            @{
        */

        /** Tag to make std::integral_constants boost::mpl compatible
         */
        template <class T, T V>
        struct equal_to_tag<std::integral_constant<T, V>> {
            using type = integral_c_tag;
        };

        /** Tag to make std::integral_constants boost::mpl compatible
         */
        template <class T, T V>
        struct not_equal_to_tag<std::integral_constant<T, V>> {
            using type = integral_c_tag;
        };

        /** Tag to make std::integral_constants boost::mpl compatible
         */
        template <class T, T V>
        struct less_tag<std::integral_constant<T, V>> {
            using type = integral_c_tag;
        };

        /** Tag to make std::integral_constants boost::mpl compatible
         */
        template <class T, T V>
        struct less_equal_tag<std::integral_constant<T, V>> {
            using type = integral_c_tag;
        };

        /** Tag to make std::integral_constants boost::mpl compatible
         */
        template <class T, T V>
        struct greater_tag<std::integral_constant<T, V>> {
            using type = integral_c_tag;
        };

        /** Tag to make std::integral_constants boost::mpl compatible
         */
        template <class T, T V>
        struct greater_equal_tag<std::integral_constant<T, V>> {
            using type = integral_c_tag;
        };

        /** Tag to make std::integral_constants boost::mpl compatible
         */
        template <class T, T V>
        struct plus_tag<std::integral_constant<T, V>> {
            using type = integral_c_tag;
        };

        /** Tag to make std::integral_constants boost::mpl compatible
         */
        template <class T, T V>
        struct minus_tag<std::integral_constant<T, V>> {
            using type = integral_c_tag;
        };
        /** @} */
        /** @} */
        /** @} */
    } // namespace mpl
} // namespace boost
