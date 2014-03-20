#pragma once

namespace gridtools {
    struct computation {
        virtual void ready() = 0;
        virtual void steady() = 0;
        virtual void finalize() = 0;
        virtual void run() = 0;
    };

} //namespace gridtools
