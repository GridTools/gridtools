/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
            typename boost::mpl::fold< T, boost::mpl::vector0<>, combine< boost::mpl::_2, boost::mpl::_1 > >::type type;
    };
}
