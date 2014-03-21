#pragma once

#include <iostream>
#include <boost/mpl/min_max.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/reverse_fold.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/static_assert.hpp>

namespace gridtools { 
    /**
     * Class to specify access ranges for stencil functions
     */
    template <int _iminus=0, int _iplus=0,
              int _jminus=0, int _jplus=0>
    struct range {
        typedef boost::mpl::int_<_iminus> iminus;
        typedef boost::mpl::int_<_iplus> iplus;
        typedef boost::mpl::int_<_jminus> jminus;
        typedef boost::mpl::int_<_jplus> jplus;
    };

    /**
     * Output operator for ranges - for debug purposes
     * 
     * @param s The ostream
     * @param n/a Arguemnt to deduce range type
     * @return The ostream
     */
    template <int i1, int i2, int i3, int i4>
    std::ostream& operator<<(std::ostream &s, range<i1,i2,i3,i4>) {
        return s << "[" 
                 << i1 << ", "
                 << i2 << ", "
                 << i3 << ", "
                 << i4 << "]";
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
    template <int I, int J, int K, int L>
    struct is_range<range<I,J,K,L> >
      : boost::true_type
    {};

    /**
     * Metafunction taking two ranges and yielding a range containing them
     */
    template <typename range1,
              typename range2>
    struct enclosing_range {
        BOOST_STATIC_ASSERT(is_range<range1>::type::value);
        BOOST_STATIC_ASSERT(is_range<range2>::type::value);

        typedef range<boost::mpl::min<typename range1::iminus, typename range2::iminus>::type::value,
                      boost::mpl::max<typename range1::iplus,  typename range2::iplus>::type::value,
                      boost::mpl::min<typename range1::jminus, typename range2::jminus>::type::value,
                      boost::mpl::max<typename range1::jplus,  typename range2::jplus>::type::value
                      > type;
    };

    /**
     * Metafunction taking two ranges and yielding a range which is the extension of one another
     */
    template <typename range1,
              typename range2>
    struct sum_range {
        BOOST_STATIC_ASSERT(is_range<range1>::type::value);
        BOOST_STATIC_ASSERT(is_range<range2>::type::value);

        typedef range<boost::mpl::plus<typename range1::iminus, typename range2::iminus>::type::value,
                      boost::mpl::plus<typename range1::iplus,  typename range2::iplus>::type::value,
                      boost::mpl::plus<typename range1::jminus, typename range2::jminus>::type::value,
                      boost::mpl::plus<typename range1::jplus,  typename range2::jplus>::type::value
                      > type;
    };
} // namespace gridtools
