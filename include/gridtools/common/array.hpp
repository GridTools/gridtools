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
/**
@file
@brief Implementation of an array class
*/
#include "defs.hpp"
#include "gt_assert.hpp"
#include "host_device.hpp"
#include <algorithm>
#include <type_traits>

namespace gridtools {

    /** \defgroup common Common Shared Utilities
        @{
     */

    /** \defgroup array Array
        @{
    */

    namespace impl_ {
        template <typename T, std::size_t N>
        struct array_traits {
            using type = T[N];
            static constexpr GT_FUNCTION bool assert_range(size_t i) { return i < N; }
        };

        template <typename T>
        struct array_traits<T, 0> {
            using type = T[1]; // maybe use implementation from std::array instead?
            static constexpr GT_FUNCTION bool assert_range(size_t) { return false; }
        };
    } // namespace impl_

    template <typename T>
    struct is_array;

    /** \brief A class equivalent to std::array but enabled for GridTools use

        \tparam T Value type of the array
        \tparam B Size of the array
     */
    template <typename T, size_t D>
    class array {
        typedef array<T, D> type;

      public:
        // we make the members public to make this class an aggregate
        typename impl_::array_traits<T, D>::type m_array;

        typedef T value_type;

        GT_FUNCTION
        T const *begin() const { return &m_array[0]; }

        GT_FUNCTION
        T *begin() { return &m_array[0]; }

        GT_FUNCTION
        T const *end() const { return &m_array[D]; }

        GT_FUNCTION
        T *end() { return &m_array[D]; }

        GT_FUNCTION
        constexpr const T *data() const noexcept { return m_array; }
        GT_FUNCTION
        T *data() noexcept { return m_array; }

        GT_FUNCTION
        constexpr T const &operator[](size_t i) const { return m_array[i]; }

        template <size_t I>
        GT_FUNCTION constexpr T get() const {
            GRIDTOOLS_STATIC_ASSERT((I < D), GT_INTERNAL_ERROR_MSG("Array out of bounds access."));
            return m_array[I];
        }

        GT_FUNCTION
        T &operator[](size_t i) {
            assert((impl_::array_traits<T, D>::assert_range(i)));
            return m_array[i];
        }

        template <typename A>
        GT_FUNCTION array &operator=(A const &a) {
            assert(a.size() == D);
            std::copy(a.begin(), a.end(), m_array);
            return *this;
        }

        GT_FUNCTION
        static constexpr size_t size() { return D; }
    };

    // in case we need a constexpr version we need to implement a recursive one for c++11
    template <typename T, typename U, size_t D>
    GT_CXX14CONSTEXPR GT_FUNCTION bool operator==(gridtools::array<T, D> const &a, gridtools::array<U, D> const &b) {
        for (size_t i = 0; i < D; ++i) {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }

    template <typename T, typename U, size_t D>
    GT_CXX14CONSTEXPR GT_FUNCTION bool operator!=(gridtools::array<T, D> const &a, gridtools::array<U, D> const &b) {
        return !(a == b);
    }

    template <typename T>
    struct is_array : boost::mpl::false_ {};

    template <typename T, size_t D>
    struct is_array<array<T, D>> : boost::mpl::true_ {};

    template <typename Array, typename Value>
    struct is_array_of : boost::mpl::false_ {};

    template <size_t D, typename Value>
    struct is_array_of<array<Value, D>, Value> : boost::mpl::true_ {};

    template <typename T>
    struct tuple_size;

    template <typename T, size_t D>
    struct tuple_size<array<T, D>> : std::integral_constant<size_t, D> {};

    template <size_t, typename T>
    struct tuple_element;

    template <size_t I, typename T, size_t D>
    struct tuple_element<I, array<T, D>> {
        using type = T;
    };

    template <size_t I, typename T, size_t D>
    GT_FUNCTION constexpr T &get(array<T, D> &arr) noexcept {
        GRIDTOOLS_STATIC_ASSERT(I < D, "index is out of bounds");
        return arr.m_array[I];
    }

    template <size_t I, typename T, size_t D>
    GT_FUNCTION constexpr const T &get(const array<T, D> &arr) noexcept {
        GRIDTOOLS_STATIC_ASSERT(I < D, "index is out of bounds");
        return arr.m_array[I];
    }

    template <size_t I, typename T, size_t D>
    GT_FUNCTION constexpr T &&get(array<T, D> &&arr) noexcept {
        GRIDTOOLS_STATIC_ASSERT(I < D, "index is out of bounds");
        return std::move(get<I>(arr));
    }

    /** @} */
    /** @} */

} // namespace gridtools
