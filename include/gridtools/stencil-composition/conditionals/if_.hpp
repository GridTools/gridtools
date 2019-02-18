/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <functional>

#include "../mss.hpp"
#include "condition.hpp"
#include "condition_tree.hpp"
/**@file*/

namespace gridtools {

    /**@brief API for specifying a boolean conditional

       \tparam Mss1 the type of the resulting multi-stage-setncil in case the condiiton is true
       \tparam Mss2 the type of the resulting multi-stage-setncil in case the condiiton is false
       \tparam Condition the type of the condition

       \param cond the runtime condition
       \param mss1_ dummy argument
       \param mss2_ dummy argument

       usage example (suppose that boolean_value is the runtime variable for the branch selection):
    @verbatim
    conditional<0>(boolean_value);

    auto comp = make_copmutation(
       if_(cond
           , make_mss(...) // true branch
           , make_mss(...) // false branch
       )
    )
    @endverbatim
    Multiple if_ statements can coexist in the same computation, and they can be arbitrarily nested
    */
    template <typename Mss1, typename Mss2, typename Condition>
    condition<Mss1, Mss2, Condition> if_(Condition cond, Mss1 const &mss1_, Mss2 const &mss2_) {
        GT_STATIC_ASSERT((std::is_convertible<Condition, std::function<bool()>>::value),
            "Condition should be nullary boolean functor.");
        GT_STATIC_ASSERT((is_condition_tree_of<Mss1, is_mss_descriptor>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_condition_tree_of<Mss2, is_mss_descriptor>::value), GT_INTERNAL_ERROR);
        return condition<Mss1, Mss2, Condition>{cond, mss1_, mss2_};
    }
} // namespace gridtools
