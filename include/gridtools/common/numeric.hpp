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

#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>

namespace gridtools {
#if __cplusplus < 201703
    template <class T>
    constexpr T gcd(T m, T n) {
        static_assert(!std::is_signed<T>::value, "");
        return n == 0 ? m : gcd<T>(n, m % n);
    }

    template <class T, class U>
    constexpr std::common_type_t<T, U> gcd(T m, U n) {
        static_assert(std::is_integral<T>() && std::is_integral<U>(), "Arguments to gcd must be integer types");
        static_assert(!std::is_same<std::remove_cv_t<T>, bool>(), "First argument to gcd cannot be bool");
        static_assert(!std::is_same<std::remove_cv_t<U>, bool>(), "Second argument to gcd cannot be bool");
        using res_t = std::common_type_t<T, U>;
        using ures_t = std::make_unsigned_t<res_t>;
        return static_cast<res_t>(gcd(static_cast<ures_t>(std::abs(m)), static_cast<ures_t>(std::abs(n))));
    }

    template <class T, class U>
    constexpr std::common_type_t<T, U> lcm(T m, U n) {
        static_assert(std::is_integral<T>() && std::is_integral<U>(), "Arguments to lcm must be integer types");
        static_assert(!std::is_same<std::remove_cv_t<T>, bool>(), "First argument to gcd cannot be bool");
        static_assert(!std::is_same<std::remove_cv_t<U>, bool>(), "Second argument to gcd cannot be bool");
        if (m == 0 || n == 0)
            return 0;
        using res_t = std::common_type_t<T, U>;
        res_t a = std::abs(m) / gcd(m, n);
        res_t b = std::abs(n);
        assert(std::numeric_limits<res_t>::max() / a > b);
        return a * b;
    }
#else
    using std::gcd;
    using std::lcm;
#endif
} // namespace gridtools
