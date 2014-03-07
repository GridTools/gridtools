#pragma once

namespace gridtools {
    struct computation {
        virtual void setup() = 0;
        virtual void prepare() = 0;
        virtual void finalize() = 0;
        virtual void run() = 0;
    };

} //namespace gridtools
