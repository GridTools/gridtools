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

// Extracted from Andrei Alexandrescu @GoingNative2013
#pragma once
#include "common/array.hpp"
#include "common/tuple.hpp"

namespace gridtools {

    template < unsigned K, class R, class F, class Array >
    struct Expander;

    template < unsigned K, class R, class F, typename ArrayValue, size_t ArraySize >
    struct Expander< K, R, F, const array< ArrayValue, ArraySize > & > {
        typedef const array< ArrayValue, ArraySize > &array_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(array_t &&a, Us &&... args) {
            return Expander< K - 1, R, F, array_t >::expand(a, a[K - 1], args...);
        }
    };

    template < class F, class R, typename ArrayValue, size_t ArraySize >
    struct Expander< 0, R, F, const array< ArrayValue, ArraySize > & > {

        typedef const array< ArrayValue, ArraySize > &array_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(array_t &&, Us... args) {
            return F::apply(args...);
        }
    };

    template < unsigned K, class R, class F, typename... Args >
    struct Expander< K, R, F, const tuple< Args... > & > {
        typedef const tuple< Args... > &tuple_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(tuple_t &&a, Us &&... args) {
            return Expander< K - 1, R, F, tuple_t >::expand(a, a.template get< K - 1 >(), args...);
        }
    };

    template < class F, class R, typename... Args >
    struct Expander< 0, R, F, const tuple< Args... > & > {

        typedef const tuple< Args... > &tuple_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(tuple_t &&, Us... args) {
            return F::apply(args...);
        }
    };

    template < unsigned K, class R, class F, typename ExtraData, class Array >
    struct Expander_inj;

    template < unsigned K, class R, class F, typename ExtraData, typename ArrayValue, size_t ArraySize >
    struct Expander_inj< K, R, F, ExtraData, const array< ArrayValue, ArraySize > & > {
        typedef const array< ArrayValue, ArraySize > &array_t;

        template < class... Us >
        GT_FUNCTION static constexpr R expand(const ExtraData &extra_data, array_t &&a, Us &&... args) {
            return Expander_inj< K - 1, R, F, ExtraData, array_t >::expand(extra_data, a, a[K - 1], args...);
        }
    };

    template < class R, class F, typename ExtraData, typename ArrayValue, size_t ArraySize >
    struct Expander_inj< 0, R, F, ExtraData, const array< ArrayValue, ArraySize > & > {
        typedef const array< ArrayValue, ArraySize > &array_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(const ExtraData &extra_data, array_t &&, Us... args) {
            return F::apply(extra_data, args...);
        }
    };

    template < unsigned K, class R, class F, typename ExtraData, typename... TupleArgs >
    struct Expander_inj< K, R, F, ExtraData, const tuple< TupleArgs... > & > {
        typedef const tuple< TupleArgs... > &tuple_t;

        template < class... Us >
        GT_FUNCTION static constexpr R expand(const ExtraData &extra_data, tuple_t &&a, Us &&... args) {
            return Expander_inj< K - 1, R, F, ExtraData, tuple_t >::expand(
                extra_data, a, a.template get< K - 1 >(), args...);
        }
        template < class... Us >
        GT_FUNCTION static constexpr R expand(ExtraData &extra_data, tuple_t &&a, Us &&... args) {
            return Expander_inj< K - 1, R, F, ExtraData, tuple_t >::expand(
                extra_data, a, a.template get< K - 1 >(), args...);
        }
    };

    template < class R, class F, typename ExtraData, typename... TupleArgs >
    struct Expander_inj< 0, R, F, ExtraData, const tuple< TupleArgs... > & > {
        typedef const tuple< TupleArgs... > &tuple_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(const ExtraData &extra_data, tuple_t &&, Us... args) {
            return F::apply(extra_data, args...);
        }

        template < class... Us >
        GT_FUNCTION static constexpr R expand(ExtraData &extra_data, tuple_t &&, Us... args) {
            return F::apply(extra_data, args...);
        }
    };

    /**
     * it expands the arguments in the Array (which should be an array or a tuple and passes the expansion
     * to the call of the functor Fn::apply(...)
     * @tparam ReturnType return type of the functor
     * @tparam Fn Functor containing the apply method being called with the expanded array arguments
     */
    template < typename ReturnType, typename Fn, typename Array >
    GT_FUNCTION static constexpr auto explode(const Array &a) -> ReturnType {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value || is_tuple< Array >::value), "Error: Wrong Type");
        return Expander< tuple_size< Array >::value, ReturnType, Fn, const Array & >::expand(a);
    }

    /**
     * it expands the arguments in the Array (which should be an array or a tuple and passes the expansion
     * to the call of the functor Fn::apply(...). This version accepts extra data to be passed to the functor in
     * addition to the array arguments
     * @tparam ReturnType return type of the functor
     * @tparam Fn Functor containing the apply method being called with the expanded array arguments
     * @tparam ExtraData extra data passed to the Fn::apply in addition to the expanded array, as first argument
     */
    template < typename ReturnType, typename Fn, typename Array, typename ExtraData >
    GT_FUNCTION static constexpr auto explode(const Array &a, const ExtraData &extra_data) -> ReturnType const {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value || is_tuple< Array >::value), "Error: Wrong Type");
        return Expander_inj< tuple_size< Array >::value, ReturnType, Fn, ExtraData, const Array & >::expand(
            extra_data, a);
    }

    template < typename ReturnType, typename Fn, typename Array, typename ExtraData >
    GT_FUNCTION static constexpr auto explode(const Array &a, ExtraData &extra_data) -> ReturnType {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value || is_tuple< Array >::value), "Error: Wrong Type");
        return Expander_inj< tuple_size< Array >::value, ReturnType, Fn, ExtraData, const Array & >::expand(
            extra_data, a);
    }
}
