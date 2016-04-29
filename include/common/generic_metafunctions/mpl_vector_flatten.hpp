#pragma once

#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/vector.hpp>

namespace gridtools {

    /**@brief This meta-function takes two sequences (T, V) and merges them into one sequence.
     * @tparam T first sequence
     * @tparam V second sequence
     */
    template < typename T, typename V >
    struct combine {
        typedef typename boost::mpl::fold< T, V, boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

    /**@brief This meta-function takes a sequence (T) that contains an arbitrary number of sub-sequences
     * and creates a single vector that contains all elements of all the sub-sequences.
     * @tparam T sequence of sequences
     */
    template < class T >
    struct flatten {
        typedef
            typename boost::mpl::fold< T, boost::mpl::vector<>, combine< boost::mpl::_2, boost::mpl::_1 > >::type type;
    };
}
