#pragma once

#include "host_device.h"

namespace gridtools {
    GT_FUNCTION
    inline int modulus(int __i, int __j) {
        return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
    }
} // namespace gridtools
