/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <type_traits>

namespace gridtools {

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \defgroup ispredicate Is_Predicate Predicate
        @{
    */

    /*
     * @struct is_meta_predicate
     * Check if it yields true_type or false_type
     */
    template <typename Pred, typename = void>
    struct is_meta_predicate : std::false_type {};

    template <typename Pred>
    struct is_meta_predicate<Pred, typename std::enable_if<Pred::value || true>::type> : std::true_type {};
    /** @} */
    /** @} */
    /** @} */
} // namespace gridtools
