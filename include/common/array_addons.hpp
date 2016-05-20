#pragma once
#include <common/array.hpp>

namespace gridtools {
    template < typename T, size_t D >
    std::ostream &operator<<(std::ostream &s, array< T, D > const &a) {
        s << " {  ";
        for (int i = 0; i < D - 1; ++i) {
            s << a[i] << ", ";
        }
        s << a[D - 1] << "  } ";

        return s;
    }

} // namespace gridtools

template < typename T, typename U, size_t D >
bool operator==(gridtools::array< T, D > const a, gridtools::array< U, D > const b) {
    gridtools::array< T, D > a0 = a;
    gridtools::array< U, D > b0 = b;
    std::sort(a0.begin(), a0.end());
    std::sort(b0.begin(), b0.end());
    return std::equal(a0.begin(), a0.end(), b0.begin());
}
