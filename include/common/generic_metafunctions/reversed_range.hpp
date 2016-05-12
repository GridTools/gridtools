#pragma once
namespace gridtools{
    /**@brief alternative to boost::mlpl::range_c, which defines an extensible sequence (mpl vector) of integers of length End-Start, with step 1, in decreasing order*/
    template<typename T, T Start, T End>
    struct reversed_range{
        typedef typename boost::mpl::reverse_fold<
            boost::mpl::range_c<T, Start, End>
            , boost::mpl::vector_c<T>
            , boost::mpl::push_back<boost::mpl::_1,
                                    boost::mpl::_2>
            >::type type;
    };
}
