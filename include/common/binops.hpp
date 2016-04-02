#pragma once

namespace gridtools {
    namespace binop {
        struct sum {
            template < typename Type >
            GT_FUNCTION Type operator()(Type const &x, Type const &y) const {
                return x + y;
            }
        };

        struct prod {
            template < typename Type >
            GT_FUNCTION Type operator()(Type const &x, Type const &y) const {
                return x * y;
            }
        };
    }
}
