#pragma once
namespace gridtools{
    /**@brief alternative to boost::mlpl::range_c, which defines an extensible sequence (mpl vector) of integers of length End-Start, with step 1, in decreasing order*/
    template<int_t Start, int_t End>
    struct reversed_range{
        typedef typename boost::mpl::reverse_fold<
            boost::mpl::range_c<int_t, Start, End>//numeration from 0
            , boost::mpl::vector_c<int_t>
            , boost::mpl::push_back<boost::mpl::_1,
                                    boost::mpl::_2>
            >::type type;
    };
}
