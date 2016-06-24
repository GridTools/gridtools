/**
 * @file provides functionality to call a lambda that accepts multiple arguments or a variadic pack,
 * passing a user tuple, array or object packing all arguments
 *
 * Usage example:
 *
 * struct Functor{
 *
 *   template<typename ... UInt>
 *   static bool apply(float initial_data, UInt ... args)
 *   {...}
 * };
 *
 * tuple< int, float, unsigned short > a(-35, 23.3, 9);
 * bool result = explode< bool, Functor >(a, (float)0.0);
 */

// Extracted from Andrei Alexandrescu @GoingNative2013
#pragma once
#include "common/array.hpp"
#include "common/tuple.hpp"

#ifdef CXX11_ENABLED

namespace gridtools {

    template < unsigned K, class R, class F, class Array >
    struct expander;

    template < unsigned K, class R, class F, typename ArrayValue, size_t ArraySize >
    struct expander< K, R, F, const array< ArrayValue, ArraySize > & > {
        typedef const array< ArrayValue, ArraySize > &array_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(array_t &&a, Us &&... args) {
            return expander< K - 1, R, F, array_t >::expand(a, a[K - 1], args...);
        }
    };

    template < class F, class R, typename ArrayValue, size_t ArraySize >
    struct expander< 0, R, F, const array< ArrayValue, ArraySize > & > {

        typedef const array< ArrayValue, ArraySize > &array_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(array_t &&, Us... args) {
            return F::apply(args...);
        }
    };

    template < unsigned K, class R, class F, typename... Args >
    struct expander< K, R, F, const tuple< Args... > & > {
        typedef const tuple< Args... > &tuple_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(tuple_t &&a, Us &&... args) {
            return expander< K - 1, R, F, tuple_t >::expand(a, a.template get< K - 1 >(), args...);
        }
    };

    template < class F, class R, typename... Args >
    struct expander< 0, R, F, const tuple< Args... > & > {

        typedef const tuple< Args... > &tuple_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(tuple_t &&, Us... args) {
            return F::apply(args...);
        }
    };

    template < unsigned K, class R, class F, typename ExtraData, class Array >
    struct expander_inj;

    template < unsigned K, class R, class F, typename ExtraData, typename ArrayValue, size_t ArraySize >
    struct expander_inj< K, R, F, ExtraData, const array< ArrayValue, ArraySize > & > {
        typedef const array< ArrayValue, ArraySize > &array_t;

        template < class... Us >
        GT_FUNCTION static constexpr R expand(const ExtraData &extra_data, array_t &&a, Us &&... args) {
            return expander_inj< K - 1, R, F, ExtraData, array_t >::expand(extra_data, a, a[K - 1], args...);
        }
    };

    template < class R, class F, typename ExtraData, typename ArrayValue, size_t ArraySize >
    struct expander_inj< 0, R, F, ExtraData, const array< ArrayValue, ArraySize > & > {
        typedef const array< ArrayValue, ArraySize > &array_t;
        template < class... Us >
        GT_FUNCTION static constexpr R expand(const ExtraData &extra_data, array_t &&, Us... args) {
            return F::apply(extra_data, args...);
        }
    };

    template < unsigned K, class R, class F, typename ExtraData, typename... TupleArgs >
    struct expander_inj< K, R, F, ExtraData, const tuple< TupleArgs... > & > {
        typedef const tuple< TupleArgs... > &tuple_t;

        template < class... Us >
        GT_FUNCTION static constexpr R expand(const ExtraData &extra_data, tuple_t &&a, Us &&... args) {
            return expander_inj< K - 1, R, F, ExtraData, tuple_t >::expand(
                extra_data, a, a.template get< K - 1 >(), args...);
        }
        template < class... Us >
        GT_FUNCTION static constexpr R expand(ExtraData &extra_data, tuple_t &&a, Us &&... args) {
            return expander_inj< K - 1, R, F, ExtraData, tuple_t >::expand(
                extra_data, a, a.template get< K - 1 >(), args...);
        }
    };

    template < class R, class F, typename ExtraData, typename... TupleArgs >
    struct expander_inj< 0, R, F, ExtraData, const tuple< TupleArgs... > & > {
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
        return expander< Array::n_dimensions, ReturnType, Fn, const Array & >::expand(a);
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
    GT_FUNCTION static constexpr auto explode(const Array &a, const ExtraData &extra_data) -> ReturnType {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value || is_tuple< Array >::value), "Error: Wrong Type");
        return expander_inj< Array::n_dimensions, ReturnType, Fn, ExtraData, const Array & >::expand(extra_data, a);
    }
}

#endif
