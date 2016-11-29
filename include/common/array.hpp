/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <algorithm>
#include <boost/type_traits/has_trivial_constructor.hpp>
#include <stddef.h>

#include "defs.hpp"
#include "generic_metafunctions/accumulate.hpp"
#include "gt_assert.hpp"
#include "host_device.hpp"
//#include "common/generic_metafunctions/gt_integer_sequence.hpp"

namespace gridtools {

    template < typename T >
    struct is_array;

    template < typename T, size_t D >
    class array {
        typedef array< T, D > type;
        static const uint_t _size = (D > 0) ? D : 1;

        // we make the members public to make this class an aggregate
      public:
        T _array[_size];

        typedef T value_type;
        static const size_t n_dimensions = D;

#ifdef CXX11_ENABLED

        // TODO provide a constexpr version
        T operator*(type &other) {
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

#else
        GT_FUNCTION
        array() {}

        // TODO provide a BOOST PP implementation for this
        GT_FUNCTION
        array(T const &i) : _array() {
            GRIDTOOLS_STATIC_ASSERT((!is_array< T >::value), "internal error");
            const_cast< typename boost::remove_const< T >::type * >(_array)[0] = i;
        }
        GT_FUNCTION
        array(T const &i, T const &j) : _array() {
            const_cast< typename boost::remove_const< T >::type * >(_array)[0] = i;
            const_cast< typename boost::remove_const< T >::type * >(_array)[1] = j;
        }
        GT_FUNCTION
        array(T const &i, T const &j, T const &k) : _array() {
            const_cast< typename boost::remove_const< T >::type * >(_array)[0] = i;
            const_cast< typename boost::remove_const< T >::type * >(_array)[1] = j;
            const_cast< typename boost::remove_const< T >::type * >(_array)[2] = k;
        }
        GT_FUNCTION
        array(T const &i, T const &j, T const &k, T const &l) : _array() {
            const_cast< typename boost::remove_const< T >::type * >(_array)[0] = i;
            const_cast< typename boost::remove_const< T >::type * >(_array)[1] = j;
            const_cast< typename boost::remove_const< T >::type * >(_array)[2] = k;
            const_cast< typename boost::remove_const< T >::type * >(_array)[3] = l;
        }
        GT_FUNCTION
        array(T const &i, T const &j, T const &k, T const &l, T const &p) : _array() {
            const_cast< typename boost::remove_const< T >::type * >(_array)[0] = i;
            const_cast< typename boost::remove_const< T >::type * >(_array)[1] = j;
            const_cast< typename boost::remove_const< T >::type * >(_array)[2] = k;
            const_cast< typename boost::remove_const< T >::type * >(_array)[3] = l;
            const_cast< typename boost::remove_const< T >::type * >(_array)[4] = p;
        }

        // TODO provide a BOOST PP implementation for this (so ugly :-()
        GT_FUNCTION
        array(array< T, 1 > const &other) : _array() { _array[0] = other[0]; }
        GT_FUNCTION
        array(array< T, 2 > const &other) : _array() {
            _array[0] = other[0];
            _array[1] = other[1];
        }
        GT_FUNCTION
        array(array< T, 3 > const &other) : _array() {
            _array[0] = other[0];
            _array[1] = other[1];
            _array[2] = other[2];
        }
        GT_FUNCTION
        array(array< T, 4 > const &other) : _array() {
            _array[0] = other[0];
            _array[1] = other[1];
            _array[2] = other[2];
            _array[3] = other[3];
        }
        GT_FUNCTION
        array(array< T, 5 > const &other) : _array() {
            _array[0] = other[0];
            _array[1] = other[1];
            _array[2] = other[2];
            _array[3] = other[3];
            _array[4] = other[4];
        }
#endif

        GT_FUNCTION
        T const *begin() const { return &_array[0]; }

        GT_FUNCTION
        T *begin() { return &_array[0]; }

        GT_FUNCTION
        T const *end() const { return &_array[_size]; }

        GT_FUNCTION
        T *end() { return &_array[_size]; }

        GT_FUNCTION
        T *data() const { return _array; }

        GT_FUNCTION
        constexpr T const &operator[](size_t i) const {
            // assert((i < _size));
            return _array[i];
        }

        GT_FUNCTION
        T &operator[](size_t i) {
            assert((i < _size));
            return _array[i];
        }

        template < typename A >
        GT_FUNCTION array &operator=(A const &a) {
            assert(a.size() == _size);
            std::copy(a.begin(), a.end(), _array);
            return *this;
        }

        GT_FUNCTION
        static constexpr size_t size() { return _size; }
    };

    template < typename T >
    struct is_array : boost::mpl::false_ {};

    template < typename T, size_t D >
    struct is_array< array< T, D > > : boost::mpl::true_ {};

    template < typename Array, typename Value >
    struct is_array_of : boost::mpl::false_ {};

    template < size_t D, typename Value >
    struct is_array_of< array< Value, D >, Value > : boost::mpl::true_ {};

} // namespace gridtools
