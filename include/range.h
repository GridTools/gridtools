#pragma once

#include <iostream>
#include <boost/mpl/min_max.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/reverse_fold.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/static_assert.hpp>

template <int _iminus=0, int _iplus=0,
          int _jminus=0, int _jplus=0>
struct range {
    typedef typename boost::mpl::int_<_iminus> iminus;
    typedef typename boost::mpl::int_<_iplus> iplus;
    typedef typename boost::mpl::int_<_jminus> jminus;
    typedef typename boost::mpl::int_<_jplus> jplus;
    //typedef range<_iminus, _iplus, _jminus, _jplus> type;
    //operator int() {return 0;}
};

template <int i1, int i2, int i3, int i4>
std::ostream& operator<<(std::ostream &s, range<i1,i2,i3,i4>) {
    return s << "[" 
             << i1 << ", "
             << i2 << ", "
             << i3 << ", "
             << i4 << "]";
}

template <typename T>
struct is_range {
    typedef boost::false_type type;
};

template <int I, int J, int K, int L>
struct is_range<range<I,J,K,L> > {
    typedef boost::true_type type;
};

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

// template <typename list_of_ranges>
// struct prefix_on_ranges {

//     template <typename t_list, typename t_range>
//     struct state {
//         typedef t_list list;
//         typedef t_range range;
//     };

//     template <typename previous_state, typename current_element>
//     struct update_state {
//         typedef typename sum_range<typename previous_state::range,
//                                    current_element>::type new_range;
//         typedef typename boost::mpl::push_front<typename previous_state::list, new_range>::type new_list;
//         typedef state<new_list, new_range> type;
//     };

//     typedef typename boost::mpl::reverse_fold<
//         list_of_ranges,
//         state<boost::mpl::vector<>, range<0,0,0,0> >,
//         update_state<boost::mpl::_1, boost::mpl::_2> >::type final_state;

//     typedef typename final_state::list type;
// };

