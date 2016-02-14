#pragma once
#include "case_type.hpp"

namespace gridtools {
    template<typename T, typename Mss>
    case_type<T, Mss> case_(T val_, Mss mss_)
    {
        return case_type<T, Mss>(val_, mss_);
    }

    template<typename Mss>
    default_type<Mss> default_(Mss mss_)
    {
        return default_type<Mss>(mss_);
    }
} //namespace gridtools
