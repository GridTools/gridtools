/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/tuple.hpp>

namespace gridtools {
    namespace nvcc_workarounds {

        // see https://github.com/GridTools/gridtools/issues/1766
        template <class T>
        constexpr auto make_1_tuple(T &&t) {
            return tuple(std::forward<T>(t));
        }
    } // namespace nvcc_workarounds
} // namespace gridtools
