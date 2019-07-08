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
/**
@file
@brief Implementation of an array class
*/
#include <algorithm>
#include <iterator>
#include <type_traits>

#include "../meta/id.hpp"
#include "../meta/macros.hpp"
#include "../meta/repeat.hpp"
#include "defs.hpp"
#include "generic_metafunctions/const_ref.hpp"
#include "generic_metafunctions/utility.hpp"
#include "gt_assert.hpp"
#include "host_device.hpp"

namespace gridtools {

    /** \defgroup common Common Shared Utilities
        @{
     */

    /** \addtogroup array Array
        @{
    */

    /** \brief A class equivalent to std::array but enabled for GridTools use

        \tparam T Value type of the array
        \tparam B Size of the array
     */
    template <typename T, size_t D>
    class array {
      public:
        // we make the members public to make this class an aggregate
        T m_array[D ? D : 1]; // Note: don't use a type-trait for the zero-size case, as there is a bug with defining an
                              // alias for c-arrays in CUDA 9.2 and 10.0 (see #1040)

        using value_type = T;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;
        using reference = value_type &;
        using const_reference = value_type const &;
        using pointer = value_type *;
        using const_pointer = value_type const *;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        GT_FUNCTION
        T const *begin() const { return &m_array[0]; }

        GT_FUNCTION
        T *begin() { return &m_array[0]; }

        GT_FUNCTION
        T const *end() const { return &m_array[D]; }

        GT_FUNCTION
        T *end() { return &m_array[D]; }

        GT_FUNCTION
        GT_CONSTEXPR const T *data() const noexcept { return m_array; }
        GT_FUNCTION
        T *data() noexcept { return m_array; }

        GT_FUNCTION
        GT_CONSTEXPR T const &operator[](size_t i) const { return m_array[i]; }

        GT_FUNCTION
        T &operator[](size_t i) {
            assert(i < D);
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

    namespace array_impl_ {
        template <class... Ts>
        struct deduce_array_type : std::common_type<Ts...> {};

        template <>
        struct deduce_array_type<> {
            using type = meta::lazy::id<void>;
        };

        struct from_types_f {
            template <class... Ts>
            using apply = array<typename deduce_array_type<Ts...>::type, sizeof...(Ts)>;
        };

        struct getter {
            template <size_t I, typename T, size_t D>
            static GT_FUNCTION GT_CONSTEXPR T &get(array<T, D> &arr) noexcept {
                GT_STATIC_ASSERT(I < D, "index is out of bounds");
                return arr.m_array[I];
            }

            template <size_t I, typename T, size_t D>
            static GT_FUNCTION GT_CONSTEXPR const_ref<T> get(const array<T, D> &arr) noexcept {
                GT_STATIC_ASSERT(I < D, "index is out of bounds");
                return arr.m_array[I];
            }

            template <size_t I, typename T, size_t D>
            static GT_FUNCTION GT_CONSTEXPR T get(array<T, D> &&arr) noexcept {
                GT_STATIC_ASSERT(I < D, "index is out of bounds");
                return wstd::move(arr.m_array[I]);
            }
        };
    } // namespace array_impl_

    template <typename T, size_t D>
    meta::repeat_c<D, T> tuple_to_types(array<T, D> const &);

    template <typename T, size_t D>
    array_impl_::from_types_f tuple_from_types(array<T, D> const &);

    template <typename T, size_t D>
    array_impl_::getter tuple_getter(array<T, D> const &);

    // in case we need a constexpr version we need to implement a recursive one for c++11
    template <typename T, typename U, size_t D>
    GT_CONSTEXPR GT_FUNCTION bool operator==(gridtools::array<T, D> const &a, gridtools::array<U, D> const &b) {
        for (size_t i = 0; i < D; ++i) {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }

    template <typename T, typename U, size_t D>
    GT_CONSTEXPR GT_FUNCTION bool operator!=(gridtools::array<T, D> const &a, gridtools::array<U, D> const &b) {
        return !(a == b);
    }

    template <typename T>
    struct is_array : std::false_type {};

    template <typename T, size_t D>
    struct is_array<array<T, D>> : std::true_type {};

    template <typename Array, typename Value>
    struct is_array_of : std::false_type {};

    template <size_t D, typename Value>
    struct is_array_of<array<Value, D>, Value> : std::true_type {};

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
    GT_FUNCTION GT_CONSTEXPR T &get(array<T, D> &arr) noexcept {
        GT_STATIC_ASSERT(I < D, "index is out of bounds");
        return arr.m_array[I];
    }

    template <size_t I, typename T, size_t D>
    GT_FUNCTION GT_CONSTEXPR const_ref<T> get(const array<T, D> &arr) noexcept {
        GT_STATIC_ASSERT(I < D, "index is out of bounds");
        return arr.m_array[I];
    }

    template <size_t I, typename T, size_t D>
    GT_FUNCTION GT_CONSTEXPR T get(array<T, D> &&arr) noexcept {
        GT_STATIC_ASSERT(I < D, "index is out of bounds");
        return wstd::move(get<I>(arr));
    }

    /** @} */
    /** @} */

} // namespace gridtools
