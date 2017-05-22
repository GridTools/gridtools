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

#include <array>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/plus.hpp>

#include "../../common/gt_assert.hpp"
#include "../../common/variadic_pack_metafunctions.hpp"
#include "../../common/generic_metafunctions/repeat_template.hpp"

namespace gridtools {

    /**
     *  @brief A class that is used to pass halo information to the storage info.
     *  E.g., Lets say we want to retrieve a storage_info instance with a halo
     *  of 2 in I and J direction. We have to pass following type to the storage-facility
     *  halo<2,2,0>. The I and J dimensions of the storage will be extended by 2
     *  in + and - direction.
     *  @tparam N variadic list of halo sizes
     */
    template < uint_t... N >
    struct halo {

        /**
         * @brief member function used to query the halo size of a given dimension
         * @tparam V Dimension or coordinate to query
         * @return halo size
         */
        template < uint_t V >
        static constexpr uint_t at() {
            GRIDTOOLS_STATIC_ASSERT(
                (V < sizeof...(N)), GT_INTERNAL_ERROR_MSG("Out of bounds access in halo type discovered."));
            return get_value_from_pack(V, N...);
        }

        /**
         * @brief member function used to query the halo size of a given dimension
         * @param V Dimension or coordinate to query
         * @return halo size
         */
        static constexpr uint_t at(uint_t V) { return get_value_from_pack(V, N...); }

        /**
         * @brief member function used to query the number of dimensions. E.g., a halo
         * type with 3 entries cannot be passed to a <3 or >3 dimensional storage_info.
         * @return number of dimensions
         */
        static constexpr uint_t size() { return sizeof...(N); }
    };

    /**
     *  @brief Used to generate a zero initialzed halo. Used as a default value for storage info halo.
     */
    template < uint_t Cnt >
    using zero_halo = typename repeat_template_c< 0, Cnt, halo >::type;

    /* used to check if a given type is a halo type */
    template < typename T >
    struct is_halo : boost::mpl::false_ {};

    template < uint_t... N >
    struct is_halo< halo< N... > > : boost::mpl::true_ {};
}
