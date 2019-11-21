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
        static constexpr int max_arg = constexpr_max(Args...);

        /** @brief Length of layout map excluding masked dimensions. */
        static constexpr std::size_t unmasked_length = meta::length<unmasked_args>::value;
        /** @brief Total length of layout map, including masked dimensions. */
        static constexpr std::size_t masked_length = sizeof...(Args);

        static_assert(unmasked_arg_sum == unmasked_length * (unmasked_length - 1) / 2,
            GT_INTERNAL_ERROR_MSG("Layout map args must not contain any holes (e.g., layout_map<3,1,0>)."));

        /** @brief Get the position of the element with value `i` in the layout map. */
        GT_FUNCTION static constexpr std::size_t find(int i) { return get_index_of_element_in_pack(0, i, Args...); }

        /** @brief Get the value of the element at position `I` in the layout map. */
        GT_FUNCTION static constexpr int at(std::size_t i) { return get_value_from_pack(i, Args...); }
    };
    /** @} */
    /** @} */
} // namespace gridtools
