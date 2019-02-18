/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup allmeta Generic Metafunctions

        Set of generic funcionalities to deal with types.
        @{
        \defgroup staticif Static If
        @{
    */

    /** method replacing the operator ? which selects a branch at compile time and
        allows to return different types whether the condition is true or false.
        The use is

        \code
        auto x = static_if<BOOL>::apply(true_val_of_typeA, false_val_of_value_B);
        \endcode

        \tparam Condition The evaluated boolean condition
    */
    template <bool Condition>
    struct static_if;

    /// \private
    template <>
    struct static_if<true> {
        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION static constexpr TrueVal &apply(TrueVal &true_val, FalseVal & /*false_val*/) {
            return true_val;
        }

        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION static constexpr TrueVal const &apply(TrueVal const &true_val, FalseVal const & /*false_val*/) {
            return true_val;
        }

        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION static void eval(TrueVal const &true_val, FalseVal const & /*false_val*/) {
            true_val();
        }
    };

    /// \private
    template <>
    struct static_if<false> {
        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION static constexpr FalseVal &apply(TrueVal & /*true_val*/, FalseVal &false_val) {
            return false_val;
        }

        template <typename TrueVal, typename FalseVal>
        GT_FUNCTION static void eval(TrueVal const & /*true_val*/, FalseVal const &false_val) {
            false_val();
        }
    };
    /** @} */
    /** @} */
    /** @} */
} // namespace gridtools
