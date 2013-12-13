#pragma once

namespace gridtools {
    inline int modulus(int __i, int __j) {
        return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
    }
} // namespace gridtools
