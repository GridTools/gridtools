/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/array.hpp"

namespace gridtools {
    namespace _gcl_internal {

        template <typename T>
        inline int access(gridtools::array<T, 2> const &index, gridtools::array<T, 2> const &size) {
            return index[0] + index[1] * size[0];
        }

        template <typename T>
        inline int access(gridtools::array<T, 3> const &index, gridtools::array<T, 3> const &size) {
            return index[0] + index[1] * size[0] + index[2] * size[0] * size[1];
        }
    } // namespace _gcl_internal
} // namespace gridtools
