#pragma once
#include "../common/generic_metafunctions/static_if.hpp"

namespace gridtools{

    /** @brief defining the padding to be added to the storage addresses  for alignment reasons

        it wraps a boost::mpl::vector of static_uints
     */
#ifdef CXX11_ENABLED
    template <uint_t ... Pad >
    struct halo{
        static const uint_t size=sizeof ... (Pad);
        template<ushort_t Coordinate>
        GT_FUNCTION
        static
#ifdef NDEBUG
        constexpr
#endif
        uint_t get(){
            GRIDTOOLS_STATIC_ASSERT((Coordinate>=0), "the halo must be a non negative number");
#ifdef PEDANTIC
            GRIDTOOLS_STATIC_ASSERT((Coordinate<sizeof ... (Pad)), "the requested coordinate is larger than the halo dimension");
#endif
#ifndef NDEBUG
            assert((Coordinate<sizeof ... (Pad)));
#endif
            return gt_get<Coordinate>::apply(Pad ...);
                // static_if< Coordinate <= sizeof ... (Pad) >::apply(boost::mpl::at_c<type, Coordinate>::type::value, 0);
        }
    };
#else
    template <uint_t Pad1, uint_t Pad2, uint_t Pad3 >
    struct halo{
        static const uint_t size=3;
        template<ushort_t Coordinate>
        GT_FUNCTION
        static constexpr uint_t get(){
            GRIDTOOLS_STATIC_ASSERT(Coordinate>=0, "the halo must be a non negative number");
            GRIDTOOLS_STATIC_ASSERT(Coordinate<3, "the halo dimension is exceeding the storage dimension");
            if(Coordinate==0)
                return Pad1;
            if(Coordinate==1)
                return Pad2;
            if(Coordinate==2)
                return Pad3;
                // static_if< Coordinate <= sizeof ... (Pad) >::apply(boost::mpl::at_c<type, Coordinate>::type::value, 0);
        }
    };
#endif

    template <typename T>
    struct is_halo : boost::mpl::false_ {};

#ifdef CXX11_ENABLED
    template <uint_t ... Pad>
    struct is_halo<halo<Pad ...> > : boost::mpl::true_ {};
#else
    template <uint_t Pad1, uint_t Pad2, uint_t Pad3>
    struct is_halo<halo<Pad1, Pad2, Pad3> > : boost::mpl::true_ {};
#endif

}//namespace gridtools
