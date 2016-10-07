#pragma once
#include <boost/mpl/if.hpp>
#include "../defs.hpp"
#include "variadic_typedef.hpp"

#ifdef CXX11_ENABLED

namespace gridtools {

    namespace impl {

        template < typename ReturnType, uint_t Idx >
        constexpr ReturnType pack_get_elem_(uint pos) {
            return ReturnType();
        }

        template < typename ReturnType, uint_t Idx, typename First, typename... ElemTypes >
        constexpr ReturnType pack_get_elem_(uint pos, First first, ElemTypes... elems) {
            return (pos == Idx) ? first : pack_get_elem_< ReturnType, Idx >(pos + 1, elems...);
        }
    }

    template < int_t Idx >
    struct pack_get_elem_null {
        template < typename... ElemTypes >
        static constexpr int apply(ElemTypes... elems) {
            return 0;
        }
    };

    template < int_t Idx >
    struct pack_get_elem_elem {
        template < typename... ElemTypes >
        static constexpr
            typename variadic_typedef< ElemTypes... >::template get_elem< Idx >::type apply(ElemTypes... elems) {
            return impl::pack_get_elem_< typename variadic_typedef< ElemTypes... >::template get_elem< Idx >::type,
                Idx >(0, elems...);
        }
    };

    template < int_t Idx >
    struct pack_get_elem {
        using type = typename boost::mpl::if_c< (Idx < 0), pack_get_elem_null< Idx >, pack_get_elem_elem< Idx > >::type;
    };
}
#endif
