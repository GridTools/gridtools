#pragma once

#include <iostream>
#include <boost/mpl/min_max.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/reverse_fold.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/push_front.hpp>

namespace gridtools {

    /**
     * Class to specify access ranges for stencil functions
     */
    template <int_t IMinus, int_t IPlus,
              int_t JMinus, int_t JPlus,
              int_t KMinus=0, int_t KPlus=0>
    struct range {
#ifndef CXX11_ENABLED
        typedef boost::mpl::int_<IMinus> iminus;
        typedef boost::mpl::int_<IPlus>  iplus;
        typedef boost::mpl::int_<JMinus> jminus;
        typedef boost::mpl::int_<JPlus>  jplus;
        typedef boost::mpl::int_<KMinus> kminus;
        typedef boost::mpl::int_<KPlus>  kplus;
#else
        typedef static_int<IMinus> iminus;
        typedef static_int<IPlus>  iplus;
        typedef static_int<JMinus> jminus;
        typedef static_int<JPlus>  jplus;
        typedef static_int<KMinus> kminus;
        typedef static_int<KPlus>  kplus;
#endif
    };

    template <typename In>
    struct is_staggered : public boost::false_type {};

#ifdef CXX11_ENABLED
    template <int_t ... Coords>
    struct staggered : public range<Coords ...> {};

    template <int_t ... Coords>
    struct is_staggered<staggered<Coords ...> > : public boost::true_type {};

#else

    template <int_t Coord1Minus, int_t Coord1Plus,
              int_t Coord2Minus, int_t Coord2Plus,
              int_t Coord3Minus=0, int_t Coord3Plus=0>
    struct staggered : public range<Coord1Minus, Coord1Plus,
                                    Coord2Minus, Coord2Plus,
                                    Coord3Minus, Coord3Plus>
    {};

    template <int_t Coord1Minus, int_t Coord1Plus,
              int_t Coord2Minus, int_t Coord2Plus,
              int_t Coord3Minus, int_t Coord3Plus>
    struct is_staggered<staggered<Coord1Minus, Coord1Plus,
                                  Coord2Minus, Coord2Plus,
                                  Coord3Minus, Coord3Plus>
                        > : public boost::true_type {};
#endif
    /**
     * Output operator for ranges - for debug purposes
     *
     * @param s The ostream
     * @param n/a Arguemnt to deduce range type
     * @return The ostream
     */
    template <int_t I1, int_t I2, int_t I3, int_t I4, int_t I5, int_t I6>
    std::ostream& operator<<(std::ostream &s, range<I1,I2,I3,I4,I5,I6>) {
        return s << "["
                 << I1 << ", "
                 << I2 << ", "
                 << I3 << ", "
                 << I4 << ", "
                 << I5 << ", "
                 << I6 << "]";
    }

    /**
     * Metafunction to check if a type is a range
     */
    template <typename T>
    struct is_range
      : boost::false_type
    {};

    /**
     * Metafunction to check if a type is a range - Specialization yielding true
     */
    template <int_t I, int_t J, int_t K, int_t L, int_t M, int_t N>
    struct is_range<range<I,J,K,L,M,N> >
      : boost::true_type
    {};


    /**
     * Metafunction to check if a type is a range - Specialization yielding true
     */
    template <typename T>
    struct is_range<const T >
        : is_range<T>
    {};

    template<typename T> struct undef_t;
    /**
     * Metafunction taking two ranges and yielding a range containing them
     */
    template <typename Range1,
              typename Range2>
    struct enclosing_range {
        BOOST_MPL_ASSERT((is_range<Range1>));
        BOOST_MPL_ASSERT((is_range<Range2>));

        typedef range<boost::mpl::min<typename Range1::iminus, typename Range2::iminus>::type::value,
                      boost::mpl::max<typename Range1::iplus,  typename Range2::iplus>::type::value,
                      boost::mpl::min<typename Range1::jminus, typename Range2::jminus>::type::value,
                      boost::mpl::max<typename Range1::jplus,  typename Range2::jplus>::type::value,
                      boost::mpl::min<typename Range1::kminus, typename Range2::kminus>::type::value,
                      boost::mpl::max<typename Range1::kplus,  typename Range2::kplus>::type::value
                      > type;
    };

    /**
     * Metafunction taking two ranges and yielding a range which is the extension of one another
     */
    template <typename Range1,
              typename Range2>
    struct sum_range {
        BOOST_MPL_ASSERT((boost::mpl::or_<is_range<Range1>, is_staggered<Range1> >));
        BOOST_MPL_ASSERT((boost::mpl::or_<is_range<Range2>, is_staggered<Range1> >));

        typedef range<boost::mpl::plus<typename Range1::iminus, typename Range2::iminus>::type::value,
                      boost::mpl::plus<typename Range1::iplus,  typename Range2::iplus>::type::value,
                      boost::mpl::plus<typename Range1::jminus, typename Range2::jminus>::type::value,
                      boost::mpl::plus<typename Range1::jplus,  typename Range2::jplus>::type::value,
                      boost::mpl::plus<typename Range1::kminus, typename Range2::kminus>::type::value,
                      boost::mpl::plus<typename Range1::kplus,  typename Range2::kplus>::type::value
                      > type;
    };

} // namespace gridtools
