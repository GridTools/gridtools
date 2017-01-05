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

#include <boost/mpl/at.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/mpl/count_if.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/greater_equal.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/vector_c.hpp>

#include "defs.hpp"
#include "variadic_pack_metafunctions.hpp"

namespace gridtools {

    template < bool... Bitmask >
    struct selector {
        const static unsigned size = sizeof...(Bitmask);
    };

    template < typename T, int First, int... Vals >
    struct get_mpl_vector
        : get_mpl_vector< typename boost::mpl::push_back< T, boost::mpl::int_< First > >::type, Vals... > {};

    template < typename T, int First >
    struct get_mpl_vector< T, First > {
        typedef typename boost::mpl::push_back< T, boost::mpl::int_< First > >::type type;
    };

    template < int... Args >
    struct layout_map {
        static_assert(sizeof...(Args) > 0, "Zero-dimensional layout makes no sense.");

        static constexpr int length = sizeof...(Args);
        typedef typename get_mpl_vector< boost::mpl::vector_c< unsigned >, Args... >::type static_layout_vector;
        static constexpr unsigned unmasked_length = boost::mpl::count_if< static_layout_vector,
            boost::mpl::greater< boost::mpl::_, boost::mpl::int_< -1 > > >::value;

        template < int I >
        GT_FUNCTION static constexpr int find() {
            static_assert((I >= 0) && (I < unmasked_length), "This index does not exist");
            return boost::mpl::find< static_layout_vector, boost::mpl::int_< I > >::type::pos::value;
        }

        GT_FUNCTION static constexpr int find(int i) { return get_index_of_element_in_pack(0, i, Args...); }

        template < int I >
        GT_FUNCTION static constexpr int at() {
            static_assert((I <= length), "Out of bounds access");
            return boost::mpl::at< static_layout_vector, boost::mpl::int_< I > >::type::value;
        }

        GT_FUNCTION static constexpr int at(int i) { return get_value_from_pack(i, Args...); }

        template < int I >
        GT_FUNCTION static constexpr int select(int const *dims) {
            return dims[at< I >()];
        }
    };
}
