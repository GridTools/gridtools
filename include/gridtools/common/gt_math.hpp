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
#include <cmath>

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {

    /** \ingroup common
        @{
        \defgroup math Mathematical Functions
        @{
    */

    /**@brief Class in substitution of std::pow, not available in CUDA*/
    template <uint_t Number>
    struct gt_pow {
        template <typename Value>
        GT_FUNCTION static Value constexpr apply(Value const &v) {
            return v * gt_pow<Number - 1>::apply(v);
        }
    };

    /**@brief Class in substitution of std::pow, not available in CUDA*/
    template <>
    struct gt_pow<0> {
        template <typename Value>
        GT_FUNCTION static Value constexpr apply(Value const &) {
            return 1.;
        }
    };

    /*
     * @brief helper function that provides a static version of std::ceil
     * @param num value to ceil
     * @return ceiled value
     */
    GT_FUNCTION constexpr int gt_ceil(float num) {
        return (static_cast<float>(static_cast<int>(num)) == num) ? static_cast<int>(num)
                                                                  : static_cast<int>(num) + ((num > 0) ? 1 : 0);
    }

    namespace math {

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER <= 1800)
        // Intel compiler produces wrong optimized code in some rare cases in stencil functors when using const
        // references as return types, so we return value types instead of references for arithmetic types

        namespace impl_ {
            template <typename Value>
            using minmax_return_type = std::conditional_t<std::is_arithmetic<Value>::value, Value, Value const &>;
        }

        template <typename Value>
        GT_FUNCTION constexpr impl_::minmax_return_type<Value> max(Value const &val0) {
            return val0;
        }

        template <typename Value, typename... OtherValues>
        GT_FUNCTION constexpr impl_::minmax_return_type<Value> max(
            Value const &val0, Value const &val1, OtherValues const &... vals) {
            return val0 > max(val1, vals...) ? val0 : max(val1, vals...);
        }

        template <typename Value>
        GT_FUNCTION constexpr impl_::minmax_return_type<Value> min(Value const &val0) {
            return val0;
        }

        template <typename Value, typename... OtherValues>
        GT_FUNCTION constexpr impl_::minmax_return_type<Value> min(
            Value const &val0, Value const &val1, OtherValues const &... vals) {
            return val0 > min(val1, vals...) ? min(val1, vals...) : val0;
        }
#else
        template <typename Value>
        GT_FUNCTION constexpr Value const &max(Value const &val0) {
            return val0;
        }

        template <typename Value>
        GT_FUNCTION constexpr Value const &max(Value const &val0, Value const &val1) {
            return val0 > val1 ? val0 : val1;
        }

        template <typename Value, typename... OtherValues>
        GT_FUNCTION constexpr Value const &max(Value const &val0, Value const &val1, OtherValues const &... vals) {
            return val0 > max(val1, vals...) ? val0 : max(val1, vals...);
        }

        template <typename Value>
        GT_FUNCTION constexpr Value const &min(Value const &val0) {
            return val0;
        }

        template <typename Value>
        GT_FUNCTION constexpr Value const &min(Value const &val0, Value const &val1) {
            return val0 > val1 ? val1 : val0;
        }

        template <typename Value, typename... OtherValues>
        GT_FUNCTION constexpr Value const &min(Value const &val0, Value const &val1, OtherValues const &... vals) {
            return val0 > min(val1, vals...) ? min(val1, vals...) : val0;
        }
#endif

#if defined(__CUDACC__) && defined(__NVCC__)
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION decltype(auto) fabs(double val) { return ::fabs(val); }

        GT_FUNCTION decltype(auto) fabs(float val) { return ::fabs(val); }

        template <typename Value>
        GT_FUNCTION decltype(auto) fabs(Value val) {
            return ::fabs((double)val);
        }
#ifndef __CUDA_ARCH__
        GT_FUNCTION_HOST decltype(auto) fabs(long double val) { return std::fabs(val); }
#else
        // long double not supported in device code
        template <typename ErrorTrigger = double>
        GT_FUNCTION_DEVICE double fabs(long double val) {
            GT_STATIC_ASSERT((sizeof(ErrorTrigger) == 0), "long double is not supported in device code");
            return 0.;
        }
#endif
#else
        using std::fabs;
#endif

#ifdef __CUDACC__
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION decltype(auto) abs(int val) { return ::abs(val); }

        GT_FUNCTION decltype(auto) abs(long val) { return ::abs(val); }

        GT_FUNCTION decltype(auto) abs(long long val) { return ::abs(val); }

        // forward to fabs
        template <typename Value>
        GT_FUNCTION decltype(auto) abs(Value val) {
            return math::fabs(val);
        }
#else
        using std::abs;
#endif

#ifdef __CUDA_ARCH__
        /**
         * Function computing the exponential
         */
        GT_FUNCTION float exp(const float x) { return ::expf(x); }

        GT_FUNCTION double exp(const double x) { return ::exp(x); }
#else
        using std::exp;
#endif

#ifdef __CUDA_ARCH__
        /**
         * Function computing the log function
         */
        GT_FUNCTION float log(const float x) { return ::logf(x); }

        GT_FUNCTION double log(const double x) { return ::log(x); }
#else
        using std::log;
#endif

#ifdef __CUDA_ARCH__
        /**
         * Function computing the power function
         */
        GT_FUNCTION float pow(const float x, const float y) { return ::powf(x, y); }

        GT_FUNCTION double pow(const double x, const double y) { return ::pow(x, y); }
#else
        using std::pow;
#endif

#ifdef __CUDA_ARCH__
        GT_FUNCTION float sqrt(const float x) { return ::sqrtf(x); }

        GT_FUNCTION double sqrt(const double x) { return ::sqrt(x); }
#else
        using std::sqrt;
#endif

#ifdef __CUDACC__
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION decltype(auto) fmod(float x, float y) { return ::fmodf(x, y); }

        GT_FUNCTION decltype(auto) fmod(double x, double y) { return ::fmod(x, y); }

#ifdef __CUDA_ARCH__
        template <typename ErrorTrigger = int>
        GT_FUNCTION double fmod(long double x, long double y) { // return value double to suppress warning
            GT_STATIC_ASSERT(sizeof(ErrorTrigger) != 0, "long double is not supported in device code");
            return -1.;
        }
#else
        GT_FUNCTION decltype(auto) fmod(long double x, long double y) { return std::fmod(x, y); }
#endif
#else
        using std::fmod;
#endif

#ifdef __CUDACC__
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION decltype(auto) trunc(float val) { return ::truncf(val); }

        GT_FUNCTION decltype(auto) trunc(double val) { return ::trunc(val); }

        template <typename Value>
        GT_FUNCTION decltype(auto) trunc(Value val) {
            return ::trunc((double)val);
        }

#ifdef __CUDA_ARCH__
        template <typename ErrorTrigger = int>
        GT_FUNCTION double trunc(long double val) { // return value double to suppress warning
            GT_STATIC_ASSERT(sizeof(ErrorTrigger) != 0, "long double is not supported in device code");
            return 1.;
        }
#else
        GT_FUNCTION decltype(auto) trunc(long double val) { return std::trunc(val); }
#endif
#else
        using std::trunc;
#endif
    } // namespace math

    /** @} */
    /** @} */
} // namespace gridtools
