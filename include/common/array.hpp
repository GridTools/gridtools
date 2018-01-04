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

#include <stddef.h>
#include <algorithm>
#include <boost/type_traits/has_trivial_constructor.hpp>

#include "defs.hpp"
#include "gt_assert.hpp"
#include "host_device.hpp"
#include "generic_metafunctions/accumulate.hpp"

namespace gridtools {

    namespace impl_ {
        template < typename T, std::size_t N >
        struct array_traits {
            using type = T[N];
            static constexpr GT_FUNCTION bool assert_range(size_t i) { return i < N; }
        };

        template < typename T >
        struct array_traits< T, 0 > {
            using type = T[1]; // maybe use implementation from std::array instead?
            static constexpr GT_FUNCTION bool assert_range(size_t) { return false; }
        };
    }

    template < typename T >
    struct is_array;

    template < typename T, size_t D >
    class array {
        typedef array< T, D > type;

      public:
        // we make the members public to make this class an aggregate
        typename impl_::array_traits< T, D >::type _array;

        typedef T value_type;

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

        GT_FUNCTION
        T const *begin() const { return &_array[0]; }

        GT_FUNCTION
        T *begin() { return &_array[0]; }

        GT_FUNCTION
        T const *end() const { return &_array[D]; }

        GT_FUNCTION
        T *end() { return &_array[D]; }

        GT_FUNCTION
        constexpr const T *data() const noexcept { return _array; }
        GT_FUNCTION
        T *data() noexcept { return _array; }

        GT_FUNCTION
        constexpr T const &operator[](size_t i) const { return _array[i]; }

        template < size_t I >
        GT_FUNCTION constexpr T get() const {
            GRIDTOOLS_STATIC_ASSERT((I < D), GT_INTERNAL_ERROR_MSG("Array out of bounds access."));
            return _array[I];
        }

        GT_FUNCTION
        T &operator[](size_t i) {
            assert((impl_::array_traits< T, D >::assert_range(i)));
            return _array[i];
        }

        template < typename A >
        GT_FUNCTION array &operator=(A const &a) {
            assert(a.size() == D);
            std::copy(a.begin(), a.end(), _array);
            return *this;
        }

        GT_FUNCTION
        static constexpr size_t size() { return D; }
    };

    template < typename T >
    struct is_array : boost::mpl::false_ {};

    template < typename T, size_t D >
    struct is_array< array< T, D > > : boost::mpl::true_ {};

    template < typename Array, typename Value >
    struct is_array_of : boost::mpl::false_ {};

    template < size_t D, typename Value >
    struct is_array_of< array< Value, D >, Value > : boost::mpl::true_ {};

    template < typename T >
    class tuple_size;

    template < typename T, size_t D >
    class tuple_size< array< T, D > > : public gridtools::static_size_t< D > {};

} // namespace gridtools
