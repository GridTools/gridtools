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
    template <int_t IMinus=0, int_t IPlus=0,
              int_t JMinus=0, int_t JPlus=0>
    struct range {
#ifndef CXX11_ENABLED
        typedef boost::mpl::int_<IMinus> iminus;
        typedef boost::mpl::int_<IPlus> iplus;
        typedef boost::mpl::int_<JMinus> jminus;
        typedef boost::mpl::int_<JPlus> jplus;
#else
        typedef static_int<IMinus> iminus;
        typedef static_int<IPlus> iplus;
        typedef static_int<JMinus> jminus;
        typedef static_int<JPlus> jplus;
#endif
    };

    template <int_t ... Coords>
    struct staggered : public range<Coords ...> {};

    template <typename In>
    struct is_staggered : public boost::false_type {};

    template <int_t ... Coords>
    struct is_staggered<staggered<Coords ...> > : public boost::true_type {};

    /**
     * Output operator for ranges - for debug purposes
     *
     * @param s The ostream
     * @param n/a Arguemnt to deduce range type
     * @return The ostream
     */
    template <int_t I1, int_t I2, int_t I3, int_t I4>
    std::ostream& operator<<(std::ostream &s, range<I1,I2,I3,I4>) {
        return s << "["
                 << I1 << ", "
                 << I2 << ", "
                 << I3 << ", "
                 << I4 << "]";
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
    template <int_t I, int_t J, int_t K, int_t L>
    struct is_range<range<I,J,K,L> >
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
                      boost::mpl::max<typename Range1::jplus,  typename Range2::jplus>::type::value
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
                      boost::mpl::plus<typename Range1::jplus,  typename Range2::jplus>::type::value
                      > type;
    };

    /**
     * Metafunction computing the union of two ranges
     */
    template <typename Range1,
              typename Range2>
    struct union_ranges {
        GRIDTOOLS_STATIC_ASSERT((is_range<Range1>::value), "Internal Error: invalid type")
        GRIDTOOLS_STATIC_ASSERT((is_range<Range2>::value), "Internal Error: invalid type")

        typedef range<
            (Range1::iminus::value < Range2::iminus::value) ? Range1::iminus::value : Range2::iminus::value,
            (Range1::iplus::value > Range2::iplus::value) ? Range1::iplus::value : Range2::iplus::value,
            (Range1::jminus::value < Range2::jminus::value) ? Range1::jminus::value : Range2::jminus::value,
            (Range1::jplus::value < Range2::jplus::value) ? Range1::jplus::value : Range2::jplus::value
        > type;
    };

} // namespace gridtools
