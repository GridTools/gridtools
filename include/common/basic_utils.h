#pragma once

#include "host_device.h"

namespace gridtools {
    GT_FUNCTION
    inline int_t modulus(int_t __i, int_t __j) {
        return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
    }
} // namespace gridtools
