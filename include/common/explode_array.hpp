/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

// Extracted from Andrei Alexandrescu @GoingNative2013
#pragma once
#include "common/array.hpp"

#ifdef CXX11_ENABLED

namespace gridtools {

    template < unsigned K, class R, class F, class Array >
    struct Expander {

        template < class... Us >
        GT_FUNCTION static R expand(Array &&a, Us &&... args) {
            return Expander< K - 1, R, F, Array >::expand(a, a[K - 1], args...);
        }
    };

    template < class F, class R, class Array >
    struct Expander< 0, R, F, Array > {

        template < class... Us >
        GT_FUNCTION static R expand(Array &&, Us... args) {
            return F::apply(args...);
        }
    };

    template < unsigned K, class R, class F, typename Inj, class Array >
    struct Expander_inj {
        template < class... Us >
        GT_FUNCTION static R expand(const Inj &inj, Array &&a, Us &&... args) {
            return Expander_inj< K - 1, R, F, Inj, Array >::expand(inj, a, a[K - 1], args...);
        }
    };

    template < class R, class F, typename Inj, class Array >
    struct Expander_inj< 0, R, F, Inj, Array > {
        template < class... Us >
        GT_FUNCTION static R expand(const Inj &inj, Array &&, Us... args) {
            return F::apply(inj, args...);
        }
    };

    template < typename ReturnType, typename Fn, typename Array >
    GT_FUNCTION static auto explode(const Array &a) -> ReturnType {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value), "Error: Wrong Type");
        return Expander< Array::n_dimensions, ReturnType, Fn, const Array & >::expand(a);
    }

    template < typename ReturnType, typename Fn, typename Array, typename Inj >
    GT_FUNCTION static auto explode(const Array &a, const Inj &inj) -> ReturnType {
        GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value), "Error: Wrong Type");
        return Expander_inj< Array::n_dimensions, ReturnType, Fn, Inj, const Array & >::expand(inj, a);
    }
}

#endif
