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
#include "../common/generic_metafunctions/static_if.hpp"
#include "../common/generic_metafunctions/gt_get.hpp"

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
