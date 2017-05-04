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
#include "../common/generic_metafunctions/static_if.hpp"
#ifdef CXX11_ENABLED
#include "../common/generic_metafunctions/gt_get.hpp"
#endif

namespace gridtools {

/** @brief defining the padding to be added to the storage addresses  for alignment reasons

    it wraps a boost::mpl::vector of static_uints
 */
#ifdef CXX11_ENABLED
    template < uint_t... Pad >
    struct halo {
        static const uint_t size = sizeof...(Pad);

        template < ushort_t Coordinate >
        GT_FUNCTION static constexpr uint_t get() {
            GRIDTOOLS_STATIC_ASSERT((Coordinate >= 0), "the halo must be a non negative number");
#ifdef PEDANTIC
            GRIDTOOLS_STATIC_ASSERT(
                (Coordinate < sizeof...(Pad)), "the requested coordinate is larger than the halo dimension");
#endif
            return (Coordinate < sizeof...(Pad)) ? gt_get< Coordinate >::apply(Pad...) : 0;
        }
    };
#else
    template < uint_t Pad1, uint_t Pad2, uint_t Pad3 >
    struct halo {
        static const uint_t s_pad1 = Pad1;
        static const uint_t s_pad2 = Pad2;
        static const uint_t s_pad3 = Pad3;
        static const uint_t size = 3;

        template < ushort_t Coordinate >
        GT_FUNCTION static constexpr uint_t get() {
            GRIDTOOLS_STATIC_ASSERT(Coordinate >= 0, "the halo must be a non negative number");
            GRIDTOOLS_STATIC_ASSERT(Coordinate < 3, "the halo dimension is exceeding the storage dimension");
            if (Coordinate == 0u)
                return Pad1;
            if (Coordinate == 1u)
                return Pad2;
            if (Coordinate == 2u)
                return Pad3;
        }
    };
#endif

    template < typename T >
    struct is_halo : boost::mpl::false_ {};

#ifdef CXX11_ENABLED
    template < uint_t... Pad >
    struct is_halo< halo< Pad... > > : boost::mpl::true_ {};
#else
    template < uint_t Pad1, uint_t Pad2, uint_t Pad3 >
    struct is_halo< halo< Pad1, Pad2, Pad3 > > : boost::mpl::true_ {};
#endif

} // namespace gridtools
