/**
@file
@brief each element of the vector is the sum of all the frevious ones. Returning an mpl vector of static_int
*/

#pragma once

namespace gridtools{


    template<typename Sequence>
    struct inclusive_scan{

        typedef
        typename boost::mpl::fold<
            boost::mpl::range_c<int, 0, boost::mpl::size<Sequence>::type::value>
            , boost::mpl::vector1< static_int<0> >
            , boost::mpl::push_back
            < boost::mpl::_1
              ,boost::mpl::plus< boost::mpl::at<Sequence, boost::mpl::_2>
                                 , boost::mpl::back< boost::mpl::_1 > >
              >
            >
        ::type
        type;
    };


    template<typename Sequence>
    struct exclusive_scan{

        typedef
        typename boost::mpl::pop_front<typename inclusive_scan<Sequence>::type >::type
        type;
    };


}//namespace gridtools
