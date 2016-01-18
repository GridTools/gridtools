#pragma once

#include <boost/mpl/fold.hpp>

namespace gridtools{
    template <typename VItemVector>
    struct v_item_to_vector {
        typedef typename boost::mpl::fold<
            VItemVector,
            boost::mpl::vector0<>,
            boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>
            >::type type;
    };

} // namespace gridtools
