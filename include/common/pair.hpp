#pragma once

namespace gridtools {
    /**
       @brief simple wrapper for a pair of types
     */
    template < typename T, typename U >
    struct pair_type {
        typedef T first;
        typedef U second;
    };

    /**
       @brief simple pair with constexpr constructor

       NOTE: can be replaced by std::pair
     */
    template < typename T1, typename T2 >
    struct pair {
        constexpr pair(T1 t1_, T2 t2_) : first(t1_), second(t2_) {}

        T1 first;
        T2 second;
    };

    template < typename T1, typename T2 >
    constexpr pair< T1, T2 > make_pair(T1 t1_, T2 t2_) {
        return pair< T1, T2 >(t1_, t2_);
    }

} // namespace gridtools
