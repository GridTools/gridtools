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

#include "dimension_fwd.hpp"
#include "accessor_fwd.hpp"

namespace gridtools {
    // metafunction that determines if a type is a valid accessor ctr argument
    template < typename T >
    struct is_accessor_ctr_args {
        typedef typename boost::mpl::or_< typename boost::is_integral< T >::type,
            typename is_dimension< T >::type >::type type;
    };

    // metafunction that determines if a variadic pack are valid accessor ctr arguments
    template < typename... Types >
    using all_accessor_ctr_args =
        typename boost::enable_if_c< accumulate(logical_and(), is_accessor_ctr_args< Types >::type::value...),
            bool >::type;

    template < typename Accessor, typename Enable = void >
    struct is_accessor_readonly : boost::mpl::false_ {};

    template < typename Accessor >
    struct is_accessor_readonly< Accessor, typename std::enable_if< Accessor::intend == enumtype::in >::type >
        : boost::mpl::true_ {};

    /* Is written is actually "can be written", since it checks if not read only.*/
    template < typename Accessor >
    struct is_accessor_written : boost::mpl::bool_< !is_accessor_readonly< Accessor >::value > {};

    template < typename Accessor >
    struct accessor_index {
        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), GT_INTERNAL_ERROR);
        typedef typename Accessor::index_t type;
    };

    namespace _impl {
        template < ushort_t ID, typename ArgsMap >
        constexpr ushort_t get_remap_accessor_id() {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< ArgsMap >::value > 0), GT_INTERNAL_ERROR);
            // check that the key type is an int (otherwise the later has_key would never find the key)
            GRIDTOOLS_STATIC_ASSERT(
                (boost::is_same<
                    typename boost::mpl::first< typename boost::mpl::front< ArgsMap >::type >::type::value_type,
                    int >::value),
                GT_INTERNAL_ERROR);

            typedef typename boost::mpl::integral_c< int, (int)ID > index_t;

            GRIDTOOLS_STATIC_ASSERT((boost::mpl::has_key< ArgsMap, index_t >::value), GT_INTERNAL_ERROR);

            return boost::mpl::at< ArgsMap, index_t >::type::value;
        }
    }
} // namespace gridtools
