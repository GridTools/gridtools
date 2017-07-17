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

#include <boost/mpl/at.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/mpl/count_if.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/greater_equal.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/vector_c.hpp>

#include "variadic_pack_metafunctions.hpp"
#include "defs.hpp"
#include "gt_assert.hpp"
#include "generic_metafunctions/variadic_to_vector.hpp"

namespace gridtools {

    template < int... Args >
    struct layout_map {
        GRIDTOOLS_STATIC_ASSERT(sizeof...(Args) > 0, GT_INTERNAL_ERROR_MSG("Zero-dimensional layout makes no sense."));

        static constexpr int masked_length = sizeof...(Args);
        typedef typename variadic_to_vector< boost::mpl::int_< Args >... >::type static_layout_vector;
        static constexpr uint_t unmasked_length = boost::mpl::count_if< static_layout_vector,
            boost::mpl::greater< boost::mpl::_, boost::mpl::int_< -1 > > >::value;

        typedef typename boost::mpl::fold< static_layout_vector,
            boost::mpl::int_< 0 >,
            boost::mpl::if_< boost::mpl::greater< boost::mpl::_2, boost::mpl::int_< -1 > >,
                                               boost::mpl::plus< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1 > >::type accumulated_arg_sum_t;
        GRIDTOOLS_STATIC_ASSERT((accumulated_arg_sum_t::value ==
                                    ((unmasked_length - 1) * (unmasked_length - 1) + (unmasked_length - 1)) / 2),
            GT_INTERNAL_ERROR_MSG("Layout map args must not contain any holes (e.g., layout_map<3,1,0>)."));

        template < int I >
        GT_FUNCTION static constexpr int find() {
            GRIDTOOLS_STATIC_ASSERT(
                (I >= 0) && (I < unmasked_length), GT_INTERNAL_ERROR_MSG("This index does not exist"));
            return boost::mpl::find< static_layout_vector, boost::mpl::int_< I > >::type::pos::value;
        }

        GT_FUNCTION static constexpr int find(int i) { return get_index_of_element_in_pack(0, i, Args...); }

        template < int I >
        GT_FUNCTION static constexpr int at() {
            static_assert((I >= 0) && (I <= masked_length), "Out of bounds access");
            return boost::mpl::at< static_layout_vector, boost::mpl::int_< I > >::type::value;
        }

        /**
           @brief Version of at that does not check the index bound.
           This is useful to check killed-dimensions. The return value
           is -1 if the access is out of bound. The interface is left
           unappealing since it is discouraged.
        */
        template < int I >
        struct at_ {
            typedef typename boost::mpl::eval_if_c<(I < masked_length && I >= 0),
                boost::mpl::at< static_layout_vector, boost::mpl::int_< I > >, 
                boost::mpl::int_<-1> >::type val_t;
            const static int_t value = val_t::value;
        };

        GT_FUNCTION static constexpr int at(int i) { return get_value_from_pack(i, Args...); }

        template < int I >
        GT_FUNCTION static constexpr int select(int const *dims) {
            return dims[at< I >()];
        }
    };

    template < typename T >
    struct is_layout_map : boost::mpl::false_ {};

    template < int... Args >
    struct is_layout_map< layout_map< Args... > > : boost::mpl::true_ {};
}
