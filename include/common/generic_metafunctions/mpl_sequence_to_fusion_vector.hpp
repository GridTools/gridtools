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

namespace gridtools {
    template < typename VItemVector >
    struct mpl_sequence_to_fusion_vector {
        typedef typename boost::mpl::fold< VItemVector,
            boost::fusion::vector0<>,
            typename boost::fusion::result_of::push_back< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

} // namespace gridtools
