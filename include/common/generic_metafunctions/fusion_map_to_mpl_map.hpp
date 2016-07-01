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
#include <boost/fusion/container/map/convert.hpp>
#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/support/pair.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/fusion/include/mpl.hpp>

namespace gridtools {

    /**
     * @struct fusion_map_to_mpl_map
     * extract an mpl map from a fusion map
     */
    template < typename FusionMap >
    struct fusion_map_to_mpl_map {
        template < typename MplMap, typename FusionPair >
        struct insert_pair_to_map {
            typedef typename boost::mpl::insert< MplMap,
                boost::mpl::pair< typename FusionPair::first_type, typename FusionPair::second_type > >::type type;
        };

        typedef typename boost::mpl::fold< FusionMap,
            boost::mpl::map0<>,
            insert_pair_to_map< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

} // namespace gridtools
