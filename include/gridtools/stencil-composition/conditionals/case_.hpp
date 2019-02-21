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
#include "../mss.hpp"
#include "case_type.hpp"
#include "condition_tree.hpp"
/**@file*/

namespace gridtools {
    /**@brief interface for specifying a case from whithin a @ref gridtools::switch_ statement
     */
    template <typename T, typename Mss>
    case_type<T, Mss> case_(T val_, Mss mss_) {
        GT_STATIC_ASSERT((is_condition_tree_of<Mss, is_mss_descriptor>::value), GT_INTERNAL_ERROR);
        return case_type<T, Mss>(val_, mss_);
    }

    /**@brief interface for specifying a default case from whithin a @ref gridtools::switch_ statement
     */
    template <typename Mss>
    default_type<Mss> default_(Mss mss_) {
        GT_STATIC_ASSERT((is_condition_tree_of<Mss, is_mss_descriptor>::value), GT_INTERNAL_ERROR);
        return default_type<Mss>(mss_);
    }
} // namespace gridtools
