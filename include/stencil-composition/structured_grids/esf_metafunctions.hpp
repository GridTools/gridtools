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
#pragma once
#include <boost/mpl/equal.hpp>
#include "esf.hpp"

namespace gridtools {

    template < typename Esf1, typename Esf2 >
    struct esf_equal {
        GRIDTOOLS_STATIC_ASSERT(
            (is_esf_descriptor< Esf1 >::value && is_esf_descriptor< Esf2 >::value), "Error: Internal Error");
        typedef static_bool< boost::is_same< typename Esf1::esf_function, typename Esf2::esf_function >::value &&
                             boost::mpl::equal< typename Esf1::args_t, typename Esf2::args_t >::value > type;
    };

    struct extract_esf_functor {
        template < typename Esf >
        struct apply {
            GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), "Error");

            typedef typename Esf::esf_function type;
        };
    };

    template < typename Esf >
    struct esf_arg_list {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), "Error");
        typedef typename Esf::esf_function::arg_list type;
    };
}
