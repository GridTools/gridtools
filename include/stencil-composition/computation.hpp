#pragma once
#include "../common/defs.hpp"
#include <string>
// //\todo this struct becomes redundant when the auto keyword is used
namespace gridtools {
    struct computation {
        virtual void ready() = 0;
        virtual void steady() = 0;
        virtual void finalize() = 0;
        virtual void run() = 0;
        virtual std::string print_meter()=0;
    };

} //namespace gridtools
