#pragma once

namespace gridtools {

#ifdef CXX11_ENABLED
    /**@brief specialization to stop the recursion*/
    template < typename... Args >
    GT_FUNCTION static constexpr bool is_variadic_pack_of(Args... args) {
        return accumulate(logical_and(), args...);
    }

#endif
} // namespace gridtools
