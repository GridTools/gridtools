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

#include "../meta/combine.hpp"
#include "../meta/filter.hpp"
#include "../meta/length.hpp"
#include "../meta/list.hpp"
#include "../meta/macros.hpp"
#include "../meta/push_back.hpp"
#include "../meta/type_traits.hpp"
#include "defs.hpp"
#include "generic_metafunctions/accumulate.hpp"
#include "gt_assert.hpp"
#include "variadic_pack_metafunctions.hpp"

namespace gridtools {

    /** \ingroup common
        @{
        \defgroup layout Layout Map
        @{
    */

    namespace _impl {
        namespace _layout_map {
            /* helper meta functions */
            template <typename Int>
            using not_negative = bool_constant<Int::value >= 0>;
            template <typename A, typename B>
            using integral_plus = std::integral_constant<int, A::value + B::value>;
        } // namespace _layout_map
    }     // namespace _impl

    template <int... Args>
    struct layout_map {
      private:
        /* list of all arguments */
        using args = meta::list<std::integral_constant<int, Args>...>;

        /* list of all unmasked (i.e. non-negative) arguments */
        using unmasked_args = meta::filter<_impl::_layout_map::not_negative, args>;

        /* sum of all unmasked arguments (only used for assertion below) */
        static constexpr int unmasked_arg_sum = meta::lazy::combine<_impl::_layout_map::integral_plus,
            meta::push_back<unmasked_args, std::integral_constant<int, 0>>>::type::value;

      public:
        /** @brief Length of layout map excluding masked dimensions. */
        static constexpr std::size_t unmasked_length = meta::length<unmasked_args>::value;
        /** @brief Total length of layout map, including masked dimensions. */
        static constexpr std::size_t masked_length = sizeof...(Args);

        GT_STATIC_ASSERT(sizeof...(Args) > 0, GT_INTERNAL_ERROR_MSG("Zero-dimensional layout makes no sense."));
        GT_STATIC_ASSERT((unmasked_arg_sum == (unmasked_length * (unmasked_length - 1)) / 2),
            GT_INTERNAL_ERROR_MSG("Layout map args must not contain any holes (e.g., layout_map<3,1,0>)."));

        /** @brief Get the position of the element with value `I` in the layout map. */
        template <int I>
        GT_FUNCTION static constexpr std::size_t find() {
            GT_STATIC_ASSERT((I >= 0) && (I < unmasked_length), GT_INTERNAL_ERROR_MSG("This index does not exist"));
            // force compile-time evaluation
            return std::integral_constant<std::size_t, find(I)>::value;
        }

        /** @brief Get the position of the element with value `i` in the layout map. */
        GT_FUNCTION static constexpr std::size_t find(int i) { return get_index_of_element_in_pack(0, i, Args...); }

        /** @brief Get the value of the element at position `I` in the layout map. */
        template <std::size_t I>
        GT_FUNCTION static constexpr int at() {
            GT_STATIC_ASSERT(I < masked_length, GT_INTERNAL_ERROR_MSG("Out of bounds access"));
            // force compile-time evaluation
            return std::integral_constant<int, at(I)>::value;
        }

        /** @brief Get the value of the element at position `I` in the layout map. */
        GT_FUNCTION static constexpr int at(std::size_t i) { return get_value_from_pack(i, Args...); }

        /**
         * @brief Version of `at` that does not check the index bound and return -1 for out of bounds indices.
         * Use the versions with bounds check if applicable.
         */
        template <std::size_t I>
        GT_FUNCTION static GT_CONSTEXPR typename std::enable_if<(I < masked_length), int>::type at_unsafe() {
            return at<I>();
        }

        template <std::size_t I>
        GT_FUNCTION static GT_CONSTEXPR typename std::enable_if<(I >= masked_length), int>::type at_unsafe() {
            return -1;
        }

        template <std::size_t I>
        GT_FUNCTION static GT_CONSTEXPR int select(int const *dims) {
            return dims[at<I>()];
        }

        /** @brief Get the maximum element value in the layout map. */
        GT_FUNCTION static constexpr int max() { return constexpr_max(Args...); }
    };

    template <typename T>
    struct is_layout_map : std::false_type {};

    template <int... Args>
    struct is_layout_map<layout_map<Args...>> : std::true_type {};

    /** @} */
    /** @} */
} // namespace gridtools
