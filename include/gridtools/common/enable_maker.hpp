/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "host_device.hpp"

namespace gridtools {
    // NVCC 11 fails to do class template deduction in the case of nested templates
    // The only puropse of this helper is to factor out the workaround against it.
    // See hymap.hpp for the usage exapmle.
    class enable_maker {
      protected:
        template <template <class...> class L>
        struct maker {
            template <class... Args>
            GT_FORCE_INLINE constexpr L<Args...> operator()(Args const &...args) const {
                return {args...};
            }
        };
    };
} // namespace gridtools
