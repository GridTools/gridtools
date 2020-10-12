/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GT_TARGET_ITERATING
// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_IMPLICIT_CAST_HPP_
#define GT_COMMON_IMPLICIT_CAST_HPP_

#include "../meta/id.hpp"
#include "defs.hpp"
#include "host_device.hpp"

#define GT_FILENAME <gridtools/common/implicit_cast.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif // GT_COMMON_IMPLICIT_CAST_HPP_
#else

namespace gridtools {
    GT_TARGET_NAMESPACE {
        /**
         * `boost::implicit_cast` clone with constexpr and target specifiers
         *
         * The use of identity creates a non-deduced form, so that the explicit template argument must be supplied
         */
        template <class T>
        GT_TARGET GT_FORCE_INLINE GT_TARGET_CONSTEXPR T implicit_cast(typename meta::lazy::id<T>::type x) {
            return x;
        }
    }
} // namespace gridtools

#endif
