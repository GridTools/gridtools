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
