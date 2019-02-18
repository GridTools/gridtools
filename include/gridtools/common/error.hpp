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

#include <assert.h>
#include <stdexcept>

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup error Error Handling
        @{
    */

    /**
     * @brief This struct is used to trigger runtime errors. The reason
     * for having a struct is simply that this element can be used in
     * constexpr functions while a simple call to e.g., std::runtime_error
     * would not compile.
     */
    struct error {

        template <typename T>
        GT_FUNCTION static T get(char const *msg) {
#ifdef __CUDA_ARCH__
            assert(false);
            return *((T volatile *)(0x0));
#else
            throw std::runtime_error(msg);
#endif
        }

        template <typename T = uint_t>
        GT_FUNCTION static constexpr T trigger(char const *msg = "Error triggered") {
            return get<T>(msg);
        }
    };

    /**
     * @brief Helper struct used to throw an error if the condition is not met.
     * Otherwise the provided result is returned. This method can be used in constexprs.
     * @tparam T return type
     * @param cond condition that should be true
     * @param res result value
     * @param msg error message if condition is not met
     */
    template <typename T>
    GT_FUNCTION constexpr T error_or_return(bool cond, T res, char const *msg = "Error triggered") {
        return cond ? res : error::trigger<T>(msg);
    }

    /** @} */
    /** @} */
} // namespace gridtools
