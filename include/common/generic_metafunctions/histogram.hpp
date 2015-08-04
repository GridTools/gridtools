/**
@file
@brief counting all the types which are repeated and contiguous, returning an mpl vector of static_int
*/

#pragma once

namespace gridtools{


    template<typename Sequence>
    struct histogram{

        // typedef typename Sequence::fuck fuck;
        typedef
        //  typename boost::mpl::pop_back<
        typename boost::mpl::fold<
            boost::mpl::range_c<int, 0, boost::mpl::size<Sequence>::value>
            , boost::mpl::vector0< >
            , boost::mpl::if_< //boost::mpl::true_
                  boost::is_same<boost::mpl::at<Sequence, boost::mpl::_2>
                                 , boost::mpl::at<Sequence
                                                  , boost::mpl::minus<
                                                        boost::mpl::_2, static_int<1> > > >
                  , boost::mpl::pop_back<
                        boost::mpl::insert<
                            boost::mpl::_1
                            , boost::mpl::prior<
                                  boost::mpl::end<boost::mpl::_1> >
                            ,  boost::mpl::plus<boost::mpl::back<boost::mpl::_1>, static_int<1> >
                            > >
                  , boost::mpl::push_back<boost::mpl::_1, static_uint<1> >
                  >  >::type
        //     >
        // ::type
        type;
    };


}//namespace gridtools
