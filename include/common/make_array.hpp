#pragma once

#include "array.hpp"

namespace gridtools {
    __host__ __device__ inline gridtools::array< int, 2 > make_array(int i1, int i2) {
        gridtools::array< int, 2 > a;
        a[0] = i1;
        a[1] = i2;
        return a;
    }
    __host__ __device__ inline gridtools::array< int, 3 > make_array(int i1, int i2, int i3) {
        gridtools::array< int, 3 > a;
        a[0] = i1;
        a[1] = i2;
        a[2] = i3;
        return a;
    }
}
