#pragma once

#include <boost/mpl/fold.hpp>

namespace gridtools {
    template < typename VItemVector >
    struct mpl_sequence_to_fusion_vector {
        typedef typename boost::mpl::fold< VItemVector,
            boost::fusion::vector<>,
            typename boost::fusion::result_of::push_back< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

} // namespace gridtools
