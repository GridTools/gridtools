/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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
        GT_FUNCTION static Value constexpr apply(Value const &v) {
            return 1.;
        }
    };

    /*
     * @brief helper function that provides a static version of std::ceil
     * @param num value to ceil
     * @return ceiled value
     */
    GT_FUNCTION constexpr static int gt_ceil(float num) {
        return (static_cast<float>(static_cast<int>(num)) == num) ? static_cast<int>(num)
                                                                  : static_cast<int>(num) + ((num > 0) ? 1 : 0);
    }

    namespace math {

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER <= 1800)
        // Intel compiler produces wrong optimized code in some rare cases in stencil functors when using const
        // references as return types, so we return value types instead of references for arithmetic types

        namespace impl_ {
            template <typename Value>
            using minmax_return_type =
                typename std::conditional<std::is_arithmetic<Value>::value, Value, Value const &>::type;
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

#ifdef __CUDACC__
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION auto fabs(double val) -> decltype(::fabs(val)) { return ::fabs(val); }

        GT_FUNCTION auto fabs(float val) -> decltype(::fabs(val)) { return ::fabs(val); }

        template <typename Value>
        GT_FUNCTION auto fabs(Value val) -> decltype(::fabs((double)val)) {
            return ::fabs((double)val);
        }
#ifndef __CUDA_ARCH__
        GT_FUNCTION_HOST auto fabs(long double val) -> decltype(std::fabs(val)) { return std::fabs(val); }
#else
        // long double not supported in device code
        template <typename ErrorTrigger = double>
        GT_FUNCTION_DEVICE double fabs(long double val) {
            GRIDTOOLS_STATIC_ASSERT((sizeof(ErrorTrigger) == 0), "long double is not supported in device code");
            return 0.;
        }
#endif
#else
        using std::fabs;
#endif

#ifdef __CUDACC__
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION auto abs(int val) -> decltype(::abs(val)) { return ::abs(val); }

        GT_FUNCTION auto abs(long val) -> decltype(::abs(val)) { return ::abs(val); }

        GT_FUNCTION auto abs(long long val) -> decltype(::abs(val)) { return ::abs(val); }

        // forward to fabs
        template <typename Value>
        GT_FUNCTION auto abs(Value val) -> decltype(math::fabs(val)) {
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
        GT_FUNCTION auto fmod(float x, float y) -> decltype(::fmodf(x, y)) { return ::fmodf(x, y); }

        GT_FUNCTION auto fmod(double x, double y) -> decltype(::fmod(x, y)) { return ::fmod(x, y); }

#ifdef __CUDA_ARCH__
        template <typename ErrorTrigger = int>
        GT_FUNCTION double fmod(long double x, long double y) { // return value double to suppress warning
            GRIDTOOLS_STATIC_ASSERT(sizeof(ErrorTrigger) != 0, "long double is not supported in device code");
            return -1.;
        }
#else
        GT_FUNCTION auto fmod(long double x, long double y) -> decltype(std::fmod(x, y)) { return std::fmod(x, y); }
#endif
#else
        using std::fmod;
#endif

#ifdef __CUDACC__
        // providing the same overload pattern as the std library
        // auto return type to ensure that we do not accidentally cast
        GT_FUNCTION auto trunc(float val) -> decltype(::truncf(val)) { return ::truncf(val); }

        GT_FUNCTION auto trunc(double val) -> decltype(::trunc(val)) { return ::trunc(val); }

        template <typename Value>
        GT_FUNCTION auto trunc(Value val) -> decltype(::trunc((double)val)) {
            return ::trunc((double)val);
        }

#ifdef __CUDA_ARCH__
        template <typename ErrorTrigger = int>
        GT_FUNCTION double trunc(long double val) { // return value double to suppress warning
            GRIDTOOLS_STATIC_ASSERT(sizeof(ErrorTrigger) != 0, "long double is not supported in device code");
            return 1.;
        }
#else
        GT_FUNCTION auto trunc(long double val) -> decltype(std::trunc(val)) { return std::trunc(val); }
#endif
#else
        using std::trunc;
#endif
    } // namespace math

    /** @} */
    /** @} */
} // namespace gridtools
