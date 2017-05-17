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

#include <stdexcept>

#include <boost/utility.hpp>
#include <boost/mpl/and.hpp>
#include <boost/type_traits.hpp>

#include "host_device.hpp"
#include "error.hpp"

namespace gridtools {

    /* get a value from a variadic pack */
    template < unsigned Size >
    struct get_value_from_pack_functor {
        template < typename First, typename... Dims >
        GT_FUNCTION static constexpr First apply(unsigned Index, First f, Dims... d) {
            return (Index) ? get_value_from_pack_functor< Size - 1 >::apply(Index - 1, d..., f) : f;
        }
    };

    template <>
    struct get_value_from_pack_functor< 0 > {
        template < typename First, typename... Dims >
        GT_FUNCTION static constexpr First apply(unsigned Index, First f, Dims... d) {
            return (Index) ? f : f;
        }
    };

    template < typename First, typename... Dims >
    GT_FUNCTION constexpr First get_value_from_pack(unsigned v, First f, Dims... d) {
        return get_value_from_pack_functor< sizeof...(Dims) >::apply(v, f, d...);
    }

    /* get a value index from a variadic pack */
    template < unsigned Size >
    struct get_index_of_element_in_pack_functor {
        template < typename First, typename... Dims >
        GT_FUNCTION static constexpr unsigned apply(unsigned Index, First needle, Dims... d) {
            return (get_value_from_pack(Index, d...) == needle)
                       ? Index
                       : get_index_of_element_in_pack_functor< Size - 1 >::apply(Index + 1, needle, d...);
        }
    };

    template <>
    struct get_index_of_element_in_pack_functor< 0 > {
        template < typename First, typename... Dims >
        GT_FUNCTION static constexpr unsigned apply(unsigned Index, First needle, Dims... d) {
            return error_or_return((get_value_from_pack(Index, d...) == needle), Index, "Element not found");
        }
    };

    template < typename First, typename... Dims >
    GT_FUNCTION constexpr unsigned get_index_of_element_in_pack(unsigned Index, First needle, Dims... d) {
        return get_index_of_element_in_pack_functor< sizeof...(Dims) >::apply(Index, needle, d...);
    }

    /* check if all given types are integral types */
    template < typename... T >
    struct is_all_integral;

    template < typename First, typename... T >
    struct is_all_integral< First, T... > {
        typedef typename is_all_integral< T... >::type rec_t;
        typedef typename boost::mpl::and_< boost::is_integral< First >, rec_t >::type type;
    };

    template <>
    struct is_all_integral<> {
        typedef boost::mpl::true_ type;
    };
}
