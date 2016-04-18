#pragma once
#include "condition.hpp"
#include "../computation_grammar.hpp"
/**@file*/

namespace gridtools {

    /**@brief API for specifying a boolean conditional

       \tparam Mss1 the type of the resulting multi-stage-setncil in case the condiiton is true
       \tparam Mss2 the type of the resulting multi-stage-setncil in case the condiiton is false
       \tparam Condition the type of the condition

       \param cond the runtime condition, must be an instance of \ref gridtools::conditional
       \param mss1_ dummy argument
       \param mss2_ dummy argument

       usage example (suppose that boolean_value is the runtime variable for the branch selection):
    @verbatim
    conditional<0>(boolean_value);

    auto comp = make_copmutation(
       if_(cond
           , make_mss(...) // true branch
           , make_mss(...) // false branch
       )
    )
    @endverbatim
    Multiple if_ statements can coexist in the same computation, and they can be arbitrarily nested
    */
    template < typename Mss1, typename Mss2, typename Condition >
    condition< Mss1, Mss2, Condition > if_(Condition cond, Mss1 const &mss1_, Mss2 const &mss2_) {
        GRIDTOOLS_STATIC_ASSERT(is_computation_token< Mss1 >::value, "wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_computation_token< Mss2 >::value, "wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_conditional< Condition >::value,
            "you have to pass to gridtools::if_ an instance of type \"conditional\" as first argument.");
        return condition< Mss1, Mss2, Condition >(cond, mss1_, mss2_);
    }
} // namespace gridtools
