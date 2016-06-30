#pragma once

namespace gridtools {
    // define result types used for meta programming
    // (yes and no need return a different sizeof result!)
    struct yes {
        char a;
    };
    struct no {
        char a[2];
    };
} // namespace gridtools
