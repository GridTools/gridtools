#pragma once
#include "switch_variable.hpp"
/**@file
*/

namespace gridtools{

    /**@brief API for a runtime switch between several multi stage stencils

       Its implementation is recursive. It creates as many boolean conditionals as
       the number of \ref gridtools::case_ defined, and it implements the switch in terms of
       \ref gridtools::if_ constructs. The unique ID which is necessary in order to define the boolean
       conditionals in this case is assigned automatically by the library (as a very large number)
     */
    template<typename Condition, typename First, typename ... Cases>
    auto
    switch_(Condition& cond_, First const& first_, Cases ... cases_) ->
        decltype(
            if_(conditional<(uint_t) - (sizeof...(Cases)+sizeof...(Cases)*Condition::index_value)>()
                ,first_.mss()
                ,switch_(cond_, cases_ ...)
                )
            )
    {
        GRIDTOOLS_STATIC_ASSERT((is_case_type<First>::value),
                                "the entries in a switch_ statement must be case_ statements");

        //save the boolean in a vector owned by the switch variable
        //allows us to modify the switch at a later stage
        cond_.push_back_condition(cond_.value()==first_.value());
        cond_.push_back_case(first_.value());
        //choose an ID which should be unique: to pick a very large number we cast a negative number to an unsigned
        short_t* cond_ptr = &cond_.m_conditions.back();
        conditional<(uint_t) - (sizeof...(Cases)+sizeof...(Cases)*Condition::index_value)> c(cond_ptr);
        return if_(c,
                   first_.mss(),
                   switch_(cond_, cases_ ...)
            );
    }

    /**@brief recursion anchor*/
    template<typename Condition, typename Default>
    // typename switch_type<Condition, Default>::type
    typename Default::mss_t
    switch_(Condition const& cond_, Default const& last_){
        GRIDTOOLS_STATIC_ASSERT((is_default_type<Default>::value),
                                "the last entry in a switch_ statement must be a default_ statement");
        return last_.mss(); //default_ value
    }

}//namespace gridtools
