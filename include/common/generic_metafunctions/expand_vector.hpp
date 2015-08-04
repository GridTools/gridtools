/**
@file
@brief expanding each entry of the input vector according to the speciifed multiplicity vector

Each entry in the input vector is repeated a number of times specified by the multiplicity vector
*/
#pragma once
namespace gridtools{
    namespace impl_{
        template <typename Vector, typename Multiplicity, typename Arg, typename Index>
        struct expand{
            typedef typename boost::mpl::fold
            < boost::mpl::range_c<int, 0, boost::mpl::at<Multiplicity, Index >::type::value >
              , Arg
              , boost::mpl::push_back
              <boost::mpl::_1, typename boost::mpl::at<Vector, Index>::type
               > >::type type;
        };
    }//namespace impl_

    template <typename Vector, typename Multiplicity>
    struct expand{
        typedef typename boost::mpl::fold
        < boost::mpl::range_c<int, 0, boost::mpl::size<Multiplicity>::type::value >
          , boost::mpl::vector0< >
          , impl_::expand<Vector, Multiplicity, boost::mpl::_1, boost::mpl::_2>
          >::type type;
    };
}//namespace gridtools
