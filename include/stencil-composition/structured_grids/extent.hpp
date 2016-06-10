#pragma once

#include <boost/mpl/min_max.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/reverse_fold.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/push_front.hpp>

namespace gridtools {

    /**
     * Class to specify access extents for stencil functions
     */
    template < int_t IMinus = 0,
        int_t IPlus = 0,
        int_t JMinus = 0,
        int_t JPlus = 0,
        int_t KMinus = 0,
        int_t KPlus = 0
#ifdef CXX11_ENABLED
        ,
        int_t... Rest
#endif
        >
    struct extent {
        typedef static_int< IMinus > iminus;
        typedef static_int< IPlus > iplus;
        typedef static_int< JMinus > jminus;
        typedef static_int< JPlus > jplus;
        typedef static_int< KMinus > kminus;
        typedef static_int< KPlus > kplus;
    };

    template < typename In >
    struct is_staggered : public boost::false_type {};

#ifdef CXX11_ENABLED
    template < int_t... Grid >
    struct staggered : public extent< Grid... > {};

    template < int_t... Grid >
    struct is_staggered< staggered< Grid... > > : public boost::true_type {};

#else

    template < int_t Coord1Minus,
        int_t Coord1Plus,
        int_t Coord2Minus,
        int_t Coord2Plus,
        int_t Coord3Minus = 0,
        int_t Coord3Plus = 0 >
    struct staggered : public extent< Coord1Minus, Coord1Plus, Coord2Minus, Coord2Plus, Coord3Minus, Coord3Plus > {};

    template < int_t Coord1Minus,
        int_t Coord1Plus,
        int_t Coord2Minus,
        int_t Coord2Plus,
        int_t Coord3Minus,
        int_t Coord3Plus >
    struct is_staggered< staggered< Coord1Minus, Coord1Plus, Coord2Minus, Coord2Plus, Coord3Minus, Coord3Plus > >
        : public boost::true_type {};
#endif
    /**
     * Output operator for extents - for debug purposes
     *
     * @param s The ostream
     * @param n/a Arguemnt to deduce extent type
     * @return The ostream
     */
    template < int_t I1, int_t I2, int_t I3, int_t I4, int_t I5, int_t I6 >
    std::ostream &operator<<(std::ostream &s, extent< I1, I2, I3, I4, I5, I6 >) {
        return s << "[" << I1 << ", " << I2 << ", " << I3 << ", " << I4 << ", " << I5 << ", " << I6 << "]";
    }

    /**
     * Metafunction to check if a type is a extent
     */
    template < typename T >
    struct is_extent : boost::false_type {};

    /**
     * Metafunction to check if a type is a extent - Specialization yielding true
     */
    template < int_t I, int_t J, int_t K, int_t L, int_t M, int_t N >
    struct is_extent< extent< I, J, K, L, M, N > > : boost::true_type {};

    /**
     * Metafunction to check if a type is a extent - Specialization yielding true
     */
    template < typename T >
    struct is_extent< const T > : is_extent< T > {};

    /**
     * Metafunction taking two extents and yielding a extent containing them
     */
    template < typename Extent1, typename Extent2 >
    struct enclosing_extent {
        BOOST_MPL_ASSERT((is_extent< Extent1 >));
        BOOST_MPL_ASSERT((is_extent< Extent2 >));

        typedef extent< boost::mpl::min< typename Extent1::iminus, typename Extent2::iminus >::type::value,
            boost::mpl::max< typename Extent1::iplus, typename Extent2::iplus >::type::value,
            boost::mpl::min< typename Extent1::jminus, typename Extent2::jminus >::type::value,
            boost::mpl::max< typename Extent1::jplus, typename Extent2::jplus >::type::value,
            boost::mpl::min< typename Extent1::kminus, typename Extent2::kminus >::type::value,
            boost::mpl::max< typename Extent1::kplus, typename Extent2::kplus >::type::value > type;
    };

    // Specializations for the case in which a range is an mpl::void_, which is used when the extent is taken from an
    // mpl::map
    template < typename Extent >
    struct enclosing_extent< Extent, boost::mpl::void_ > {
        typedef Extent type;
    };

    template < typename Extent >
    struct enclosing_extent< boost::mpl::void_, Extent > {
        typedef Extent type;
    };

    /**
     * Metafunction taking two extents and yielding a extent which is the extension of one another
     */
    template < typename Extent1, typename Extent2 >
    struct sum_extent {
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::or_< is_extent< Extent1 >, is_staggered< Extent1 > >::value), "wrong type");
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::or_< is_extent< Extent2 >, is_staggered< Extent2 > >::value), "wrong type");

        typedef extent< boost::mpl::plus< typename Extent1::iminus, typename Extent2::iminus >::type::value,
            boost::mpl::plus< typename Extent1::iplus, typename Extent2::iplus >::type::value,
            boost::mpl::plus< typename Extent1::jminus, typename Extent2::jminus >::type::value,
            boost::mpl::plus< typename Extent1::jplus, typename Extent2::jplus >::type::value,
            boost::mpl::plus< typename Extent1::kminus, typename Extent2::kminus >::type::value,
            boost::mpl::plus< typename Extent1::kplus, typename Extent2::kplus >::type::value > type;
    };

} // namespace gridtools
