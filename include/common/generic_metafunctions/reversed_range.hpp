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
namespace gridtools {
    /**@brief alternative to boost::mlpl::range_c, which defines an extensible sequence (mpl vector) of integers of
     * length End-Start, with step 1, in decreasing order*/
    template < typename T, T Start, T End >
    struct reversed_range {
        typedef typename boost::mpl::reverse_fold< boost::mpl::range_c< T, Start, End >,
            boost::mpl::vector_c< T >,
            boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };
}
