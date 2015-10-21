#pragma once
#include "../common/generic_metafunctions/static_if.hpp"

namespace gridtools{

    /** @brief defining the padding to be added to the storage addresses  for alignment reasons

        it wraps a boost::mpl::vector of static_uints
     */
    template <uint_t ... Pad >
    struct padding{
        static const uint_t size=sizeof ... (Pad);
        template<ushort_t Coordinate>
        static constexpr uint_t get(){return gt_get<Coordinate>::apply(Pad ...);
                // static_if< Coordinate <= sizeof ... (Pad) >::apply(boost::mpl::at_c<type, Coordinate>::type::value, 0);
        }
    };

    template <typename T>
    struct is_padding : boost::mpl::false_ {};

    template <uint_t ... Pad>
    struct is_padding<padding<Pad ...> > : boost::mpl::true_ {};

}//namespace gridtools
