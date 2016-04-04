#ifndef _NUMERICS_H_
#define _NUMERICS_H_

/**
@file
@brief compile-time computation of the power three.
*/

namespace gridtools {
    namespace _impl {
        /** @brief Compute 3^I at compile time*/
        template < uint_t I >
        struct static_pow3;

        template <>
        struct static_pow3< 1 > {
            static const int value = 3;
        };

        template < uint_t I >
        struct static_pow3 {
            static const int value = 3 * static_pow3< I - 1 >::value;
        };
    }
}

#endif
