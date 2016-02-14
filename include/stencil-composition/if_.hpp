#pragma once
#include "condition.hpp"

namespace gridtools{
    template <typename Mss1, typename Mss2, typename Condition>
    condition<Mss1, Mss2, Condition>
    if_(Condition const& cond, Mss1 const& mss1_, Mss2 const& mss2_){
        GRIDTOOLS_STATIC_ASSERT(is_conditional<Condition>::value,
                                "you have to pass to gridtools::if_ an instance of type \"conditional\" as first argument.");
        return condition<Mss1, Mss2, Condition>(cond, mss1_, mss2_);
    }
}//namespace gridtools
