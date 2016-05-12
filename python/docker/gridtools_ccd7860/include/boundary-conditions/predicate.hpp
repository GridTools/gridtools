#pragma once
#include "direction.hpp"

namespace gridtools {

    struct default_predicate {
        template <typename Direction>
        bool operator()(Direction) const {
            return true;
        }
    };

} // namespace gridtools
