/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#include "../../common/defs.hpp"

/**@file*/
namespace gridtools {

    /**@brief structure containing a conditional and the two branches

       This structure is the record associated to a conditional, it contains two multi-stage stencils,
       possibly containing other conditionals themselves. One branch or the other will be eventually
       executed, depending on the content of the m_value member variable.
     */
    template <typename Mss1, typename Mss2, typename Condition>
    struct condition {
        GT_STATIC_ASSERT((std::is_convertible<Condition, std::function<bool()>>::value),
            "Condition should be nullary boolean functor.");
        Condition m_value;
        Mss1 m_first;
        Mss2 m_second;
    };
} // namespace gridtools
