#pragma once

namespace gridtools {

namespace enumtype
{
    /**
       @section enumtype
       @{
       @brief The following struct defines one specific component of a field
       It contains a direction (compile time constant, specifying the ID of the component),
       and a value (runtime value, which is storing the offset in the given direction).
       As everything what is inside the enumtype namespace, the Dimension keyword is
       supposed to be used at the application interface level.
    */
    template <ushort_t Coordinate>
    struct Dimension{
        template <typename IntType>
        GT_FUNCTION
        constexpr Dimension(IntType val) : value
#if( (!defined(CXX11_ENABLED)) )
                                         (val)
#else
            {val}
#endif
        {
            GRIDTOOLS_STATIC_ASSERT(Coordinate!=0, "The coordinate values passed to the accessor start from 1");
            GRIDTOOLS_STATIC_ASSERT(Coordinate>0, "The coordinate values passed to the accessor must be positive integerts");
        }

        /**@brief Constructor*/
        GT_FUNCTION
        constexpr Dimension(Dimension const& other):value(other.value){}

        static const ushort_t direction=Coordinate;
        int_t value;

        /**@brief syntactic sugar for user interface

           overloaded operators return Index types which provide the proper Dimension object.
           Clarifying example:
           defining
           \code{.cpp}
           typedef Dimension<5>::Index t;
           \endcode
           we can use thefollowing alias
           \code{.cpp}
           t+2 <--> Dimension<5>(2)
           \endcode

         */
        struct Index{
           GT_FUNCTION
           constexpr Index(){}
            GT_FUNCTION
            constexpr Index(Index const&){}
            typedef Dimension<Coordinate> super;
        };

    private:
        Dimension();
    };

    /**Aliases for the first three dimensions (x,y,z)*/
    typedef Dimension<1> x;
    typedef Dimension<2> y;
    typedef Dimension<3> z;

    /**@}*/
} // namespace enumtype

} // namespace gridtools
