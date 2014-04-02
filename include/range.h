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
    template <int IMinus=0, int IPlus=0,
              int JMinus=0, int JPlus=0>
    struct range {
        typedef boost::mpl::int_<IMinus> iminus;
        typedef boost::mpl::int_<IPlus> iplus;
        typedef boost::mpl::int_<JMinus> jminus;
        typedef boost::mpl::int_<JPlus> jplus;
    };

    /**
     * Output operator for ranges - for debug purposes
     * 
     * @param s The ostream
     * @param n/a Arguemnt to deduce range type
     * @return The ostream
     */
    template <int I1, int I2, int I3, int I4>
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
    template <int I, int J, int K, int L>
    struct is_range<range<I,J,K,L> >
      : boost::true_type
    {};

    /**
     * Metafunction taking two ranges and yielding a range containing them
     */
    template <typename Range1,
              typename Range2>
    struct enclosing_range {
        BOOST_STATIC_ASSERT(is_range<Range1>::type::value);
        BOOST_STATIC_ASSERT(is_range<Range2>::type::value);

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
        BOOST_STATIC_ASSERT(is_range<Range1>::type::value);
        BOOST_STATIC_ASSERT(is_range<Range2>::type::value);

        typedef range<boost::mpl::plus<typename Range1::iminus, typename Range2::iminus>::type::value,
                      boost::mpl::plus<typename Range1::iplus,  typename Range2::iplus>::type::value,
                      boost::mpl::plus<typename Range1::jminus, typename Range2::jminus>::type::value,
                      boost::mpl::plus<typename Range1::jplus,  typename Range2::jplus>::type::value
                      > type;
    };
} // namespace gridtools
