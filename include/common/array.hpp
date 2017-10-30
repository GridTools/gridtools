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
@briefImplementation of an array class
*/

#include <cstddef>
#include <iterator>
#include <algorithm>
#include <stdexcept>

#include <boost/type_traits/has_trivial_constructor.hpp>

#include "defs.hpp"
#include "gt_assert.hpp"
#include "host_device.hpp"
#include "generic_metafunctions/accumulate.hpp"

namespace gridtools {

    template < typename T, ::std::size_t D >
    struct array {
        // we make the members public to make this class an aggregate
        T _array[D];

        using value_type = T;
        using size_type = ::std::size_t;
        using difference_type = ::std::ptrdiff_t;
        using reference = value_type &;
        using const_reference = const value_type &;
        using pointer = value_type *;
        using const_pointer = const value_type *;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = ::std::reverse_iterator< iterator >;
        using const_reverse_iterator = ::std::reverse_iterator< const_iterator >;

        GT_FUNCTION reference at(size_type pos) {
            if (pos >= D)
                throw ::std::out_of_range("array::at()");
            return operator[](pos);
        }
        GT_FUNCTION const_reference at(size_type pos) const {
            if (pos >= D)
                throw ::std::out_of_range("array::at() const");
            return operator[](pos);
        }

        GT_FUNCTION reference operator[](size_type i) { return _array[i]; }
        GT_FUNCTION constexpr const_reference operator[](size_type i) const { return _array[i]; }

        GT_FUNCTION reference front() { return _array[0]; }
        GT_FUNCTION constexpr const_reference front() const { return _array[0]; }

        GT_FUNCTION reference back() { return _array[D - 1]; }
        GT_FUNCTION constexpr const_reference back() const { return _array[D - 1]; }

        GT_FUNCTION T *data() noexcept { return _array; }
        GT_FUNCTION constexpr T const *data() const noexcept { return _array; }

        GT_FUNCTION iterator begin() { return &_array[0]; }
        GT_FUNCTION constexpr const_iterator begin() const { return &_array[0]; }
        GT_FUNCTION constexpr const_iterator cbegin() const { return &_array[0]; }

        GT_FUNCTION iterator end() { return &_array[D]; }
        GT_FUNCTION constexpr const_iterator end() const { return &_array[D]; }
        GT_FUNCTION constexpr const_iterator cend() const { return &_array[D]; }

        GT_FUNCTION reverse_iterator rbegin() { return {end()}; }
        GT_FUNCTION constexpr const_reverse_iterator rbegin() const { return {end()}; }
        GT_FUNCTION constexpr const_reverse_iterator crbegin() const { return {cend()}; }

        GT_FUNCTION reverse_iterator rend() { return {begin()}; }
        GT_FUNCTION constexpr const_reverse_iterator rend() const { return {begin()}; }
        GT_FUNCTION constexpr const_reverse_iterator crend() const { return {begin()}; }

        GT_FUNCTION constexpr bool empty() const noexcept { return false; }

        GT_FUNCTION constexpr size_type size() const noexcept { return D; }
        GT_FUNCTION constexpr size_type max_size() const noexcept { return D; }

        GT_FUNCTION void fill(const T &value) {
            for (T &dst : _array)
                dst = value;
        }

        GT_FUNCTION void swap(array &other) noexcept(
            noexcept(std::swap(std::declval< T & >(), std::declval< T & >()))) {
            using ::std::swap;
            for (::std::size_t i = 0; i != D; ++i)
                swap(_array[i], other._array[i]);
        }

        // non standard interface

        static const size_t n_dimensions = D;

        // TODO provide a constexpr version
        T operator*(array &other) {
            // TODO assert T is a primitive
            T result = 0;
            for (int i = 0; i < n_dimensions; ++i) {
                result += _array[i] * other[i];
            }
            return result;
        }

        array< T, D + 1 > append_dim(T const &val) const {
            array< T, D + 1 > ret;
            for (uint_t c = 0; c < D; ++c) {
                ret[c] = this->operator[](c);
            }
            ret[D] = val;
            return ret;
        }

        array< T, D + 1 > prepend_dim(T const &val) const {
            array< T, D + 1 > ret;
            for (uint_t c = 1; c <= D; ++c) {
                ret[c] = this->operator[](c - 1);
            }
            ret[0] = val;
            return ret;
        }

        template < size_t I >
        GT_FUNCTION constexpr T get() const {
            GRIDTOOLS_STATIC_ASSERT((I < n_dimensions), GT_INTERNAL_ERROR_MSG("Array out of bounds access."));
            return _array[I];
        }

        template < typename A >
        GT_FUNCTION array &operator=(A const &a) {
            assert(a.size() == D);
            std::copy(a.begin(), a.end(), _array);
            return *this;
        }
    };

    template < typename T >
    struct array< T, 0 > {
        using value_type = T;
        using size_type = ::std::size_t;
        using difference_type = ::std::ptrdiff_t;
        using reference = value_type &;
        using const_reference = const value_type &;
        using pointer = value_type *;
        using const_pointer = const value_type *;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = ::std::reverse_iterator< iterator >;
        using const_reverse_iterator = ::std::reverse_iterator< const_iterator >;

        GT_FUNCTION reference at(size_type) { throw ::std::out_of_range("array::at"); }
        GT_FUNCTION const_reference at(size_type) const { throw ::std::out_of_range("const array::at"); }

        GT_FUNCTION reference operator[](size_type i) { return fake(); }
        GT_FUNCTION constexpr const_reference operator[](size_type i) const { return fake(); }

        GT_FUNCTION reference front() { return fake(); }
        GT_FUNCTION constexpr const_reference front() const { return fake(); }

        GT_FUNCTION reference back() { return fake(); }
        GT_FUNCTION constexpr const_reference back() const { return fake(); }

        GT_FUNCTION T *data() noexcept { return nullptr; }
        GT_FUNCTION constexpr T const *data() const noexcept { return nullptr; }

        GT_FUNCTION iterator begin() { return nullptr; }
        GT_FUNCTION constexpr const_iterator begin() const { return nullptr; }
        GT_FUNCTION constexpr const_iterator cbegin() const { return nullptr; }

        GT_FUNCTION iterator end() { return nullptr; }
        GT_FUNCTION constexpr const_iterator end() const { return nullptr; }
        GT_FUNCTION constexpr const_iterator cend() const { return nullptr; }

        GT_FUNCTION reverse_iterator rbegin() { return {end()}; }
        GT_FUNCTION constexpr const_reverse_iterator rbegin() const { return {end()}; }
        GT_FUNCTION constexpr const_reverse_iterator crbegin() const { return {cend()}; }

        GT_FUNCTION reverse_iterator rend() { return {begin()}; }
        GT_FUNCTION constexpr const_reverse_iterator rend() const { return {begin()}; }
        GT_FUNCTION constexpr const_reverse_iterator crend() const { return {begin()}; }

        GT_FUNCTION constexpr bool empty() const noexcept { return true; }

        GT_FUNCTION constexpr size_type size() const noexcept { return 0; }
        GT_FUNCTION constexpr size_type max_size() const noexcept { return 0; }

        GT_FUNCTION void fill(const T &) {}

        GT_FUNCTION void swap(array &) noexcept {}

        // non standard interface

        static const size_t n_dimensions = 0;

        // TODO provide a constexpr version
        T operator*(array &other) { return {}; }

        array< T, 1 > append_dim(T const &val) const { return array< T, 1 >{{val}}; }

        array< T, 1 > prepend_dim(T const &val) const { return array< T, 1 >{{val}}; }

        template < size_t I >
        GT_FUNCTION constexpr T get() const {
            GRIDTOOLS_STATIC_ASSERT(I < 0, GT_INTERNAL_ERROR_MSG("Array out of bounds access."));
            return fake();
        }

        template < typename A >
        GT_FUNCTION array &operator=(A const &a) {
            assert(a.size() == 0);
            return *this;
        }

      private:
        static GT_FUNCTION reference fake() {
#ifdef __CUDA_ARCH__
            __shared__
#endif
                static T obj;
            return obj;
        }
    };

    template < class T, ::std::size_t N >
    GT_FUNCTION bool operator==(const array< T, N > &lhs, const array< T, N > &rhs) {
        for (::std::size_t i = 0; i != N; ++i)
            if (!(lhs[i] == rhs[i]))
                return false;
        return true;
    }
    template < class T, ::std::size_t N >
    GT_FUNCTION bool operator!=(const array< T, N > &lhs, const array< T, N > &rhs) {
        for (::std::size_t i = 0; i != N; ++i)
            if (!(lhs[i] == rhs[i]))
                return true;
        return false;
    }

    template < class T, ::std::size_t N >
    GT_FUNCTION bool operator<(const array< T, N > &lhs, const array< T, N > &rhs) {
        for (::std::size_t i = 0; i != N; ++i) {
            if (lhs[i] < rhs[i])
                return true;
            if (rhs[i] < lhs[i])
                return false;
        }
        return false;
    }

    template < class T, ::std::size_t N >
    GT_FUNCTION bool operator<=(const array< T, N > &lhs, const array< T, N > &rhs) {
        for (::std::size_t i = 0; i != N; ++i) {
            if (lhs[i] < rhs[i])
                return true;
            if (rhs[i] < lhs[i])
                return false;
        }
        return true;
    }

    template < class T, ::std::size_t N >
    GT_FUNCTION bool operator>(const array< T, N > &lhs, const array< T, N > &rhs) {
        for (::std::size_t i = 0; i != N; ++i) {
            if (rhs[i] < lhs[i])
                return true;
            if (lhs[i] < rhs[i])
                return false;
        }
        return false;
    }

    template < class T, ::std::size_t N >
    GT_FUNCTION bool operator>=(const array< T, N > &lhs, const array< T, N > &rhs) {
        for (::std::size_t i = 0; i != N; ++i) {
            if (rhs[i] < lhs[i])
                return true;
            if (lhs[i] < rhs[i])
                return false;
        }
        return true;
    }

    template <::std::size_t I, class T, ::std::size_t N >
    GT_FUNCTION constexpr T &get(array< T, N > &a) noexcept {
        static_assert(I < N, "");
        return a._array[I];
    }

    template <::std::size_t I, class T, ::std::size_t N >
    GT_FUNCTION constexpr T &&get(array< T, N > &&a) noexcept {
        static_assert(I < N, "");
        return std::move(a._array[I]);
    }

    template <::std::size_t I, class T, ::std::size_t N >
    GT_FUNCTION constexpr const T &get(const array< T, N > &a) noexcept {
        static_assert(I < N, "");
        return a._array[I];
    }

    //    TODO(anstaf): rename struct swap that is defined in data_store_field.hpp to smth. else and uncomment this.
    //    template< class T, std::size_t N >
    //    GT_FUNCTION void swap( array<T,N>& lhs, array<T,N>& rhs ) { lhs.swap(rhs); }

    template < typename T >
    struct is_array : boost::mpl::false_ {};

    template < typename T, size_t D >
    struct is_array< array< T, D > > : boost::mpl::true_ {};

    template < typename Array, typename Value >
    struct is_array_of : boost::mpl::false_ {};

    template < size_t D, typename Value >
    struct is_array_of< array< Value, D >, Value > : boost::mpl::true_ {};

} // namespace gridtools
