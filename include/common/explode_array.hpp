
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
        GT_FUNCTION static R expand(array_t &&a, Us &&... args) {
            return expander< K - 1, R, F, array_t >::expand(a, a[K - 1], args...);
        }
    };

    template < class F, class R, typename ArrayValue, size_t ArraySize >
    struct expander< 0, R, F, const array< ArrayValue, ArraySize > & > {

        typedef const array< ArrayValue, ArraySize > &array_t;
        template < class... Us >
        GT_FUNCTION static R expand(array_t &&, Us... args) {
            return F::apply(args...);
        }
    };

    template < unsigned K, class R, class F, typename... Args >
    struct expander< K, R, F, const tuple< Args... > & > {
        typedef const tuple< Args... > &tuple_t;
        template < class... Us >
        GT_FUNCTION static R expand(tuple_t &&a, Us &&... args) {
            return expander< K - 1, R, F, tuple_t >::expand(a, a.template get< K - 1 >(), args...);
        }
    };

    template < class F, class R, typename... Args >
    struct expander< 0, R, F, const tuple< Args... > & > {

        typedef const tuple< Args... > &tuple_t;
        template < class... Us >
        GT_FUNCTION static R expand(tuple_t &&, Us... args) {
            return F::apply(args...);
        }
    };

    template < unsigned K, class R, class F, typename Inj, class Array >
    struct expander_inj;

    template < unsigned K, class R, class F, typename Inj, typename ArrayValue, size_t ArraySize >
    struct expander_inj< K, R, F, Inj, const array< ArrayValue, ArraySize > & > {
        typedef const array< ArrayValue, ArraySize > &array_t;

        template < class... Us >
        GT_FUNCTION static R expand(const Inj &inj, array_t &&a, Us &&... args) {
            return expander_inj< K - 1, R, F, Inj, array_t >::expand(inj, a, a[K - 1], args...);
        }
    };

    template < class R, class F, typename Inj, typename ArrayValue, size_t ArraySize >
    struct expander_inj< 0, R, F, Inj, const array< ArrayValue, ArraySize > & > {
        typedef const array< ArrayValue, ArraySize > &array_t;
        template < class... Us >
        GT_FUNCTION static R expand(const Inj &inj, array_t &&, Us... args) {
            return F::apply(inj, args...);
        }
    };

    template < unsigned K, class R, class F, typename Inj, typename... TupleArgs >
    struct expander_inj< K, R, F, Inj, const tuple< TupleArgs... > & > {
        typedef const tuple< TupleArgs... > &tuple_t;

        template < class... Us >
        GT_FUNCTION static R expand(const Inj &inj, tuple_t &&a, Us &&... args) {
            return expander_inj< K - 1, R, F, Inj, tuple_t >::expand(inj, a, a.template get< K - 1 >(), args...);
        }
        template < class... Us >
        GT_FUNCTION static R expand(Inj &inj, tuple_t &&a, Us &&... args) {
            return expander_inj< K - 1, R, F, Inj, tuple_t >::expand(inj, a, a.template get< K - 1 >(), args...);
        }
    };

    template < class R, class F, typename Inj, typename... TupleArgs >
    struct expander_inj< 0, R, F, Inj, const tuple< TupleArgs... > & > {
        typedef const tuple< TupleArgs... > &tuple_t;
        template < class... Us >
        GT_FUNCTION static R expand(const Inj &inj, tuple_t &&, Us... args) {
            return F::apply(inj, args...);
        }

        template < class... Us >
        GT_FUNCTION static R expand(Inj &inj, tuple_t &&, Us... args) {
            return F::apply(inj, args...);
        }
    };

    template < typename ReturnType, typename Fn, typename Array >
    GT_FUNCTION static auto explode(const Array &a) -> ReturnType {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value || is_tuple< Array >::value), "Error: Wrong Type");
        return expander< Array::n_dimensions, ReturnType, Fn, const Array & >::expand(a);
    }

    template < typename ReturnType, typename Fn, typename Array, typename Inj >
    GT_FUNCTION static auto explode(const Array &a, const Inj &inj) -> ReturnType {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value || is_tuple< Array >::value), "Error: Wrong Type");
        return expander_inj< Array::n_dimensions, ReturnType, Fn, Inj, const Array & >::expand(inj, a);
    }

    template < typename ReturnType, typename Fn, typename Array, typename Inj >
    GT_FUNCTION static auto explode(const Array &a, Inj &inj) -> ReturnType {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value || is_tuple< Array >::value), "Error: Wrong Type");
        return expander_inj< Array::n_dimensions, ReturnType, Fn, Inj, const Array & >::expand(inj, a);
    }
}

#endif
