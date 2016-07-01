/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include "switch_variable.hpp"
#include "../computation_grammar.hpp"
/**@file
*/

namespace gridtools {

    /**@brief API for a runtime switch between several multi stage stencils

       Its implementation is recursive. It creates as many boolean conditionals as
       the number of \ref gridtools::case_ defined, and it implements the switch in terms of
       \ref gridtools::if_ constructs. The unique ID which is necessary in order to define the boolean
       conditionals in this case is assigned automatically by the library (as a very large number)

       \tparam Condition must be of type @ref gridtools::switch_variable
       \tparam First must be an instance of @ref gridtools::case_type
       \tparam Cases must be instances of @ref gridtools::case_type, or @ref gridtools::default_type fir the last one

       \param cond_ an instance of type @ref gridtools::switch_variable, containing the value with which to compare each
case
       \param first_ the first case
       \param cases_ the pack of cases, the last one being the default_ one

       NOTE: multiple switch statements can coexist in the same computation,
       and the arbitrary nesting of switch_ statements are supported, with the constraint that the user specifies a
       switch_variable with a different Id for each switch_. Also the same switch_variable cannot be used twice.


       example of usage:
@verbatim
switch_variable<0,int> c1(3)
switch_variable<0,int> c2(1)

auto computation_ = make_computation(
    switch_(
        c1
        , case_(1, make_mss(...))
        , case_(2, make_mss(...))
        , case_(3, make_mss(...))
        , default_(make_mss(...))
    )
);

computation->ready();
computation->steady();
computation->run(); // run the 3rd case
reset_conditional(c1, c2);
computation->run(); // run the first case
computation->finalize();
@endverbatim
    */
    template < typename Condition, typename First, typename... Cases >
    auto switch_(Condition &cond_, First const &first_, Cases const &... cases_)
        -> decltype(if_(conditional< (uint_t) - (sizeof...(Cases)), Condition::index_value >(),
            first_.mss(),
            recursive_switch(uint_t(0), cond_, cases_...))) {
        GRIDTOOLS_STATIC_ASSERT(
            (is_case_type< First >::value), "the entries in a switch_ statement must be case_ statements");

        GRIDTOOLS_STATIC_ASSERT((is_switch_variable< Condition >::value),
            "the first argument of the switch_ statement must be of switch_variable type");

        // save the boolean in a vector owned by the switch variable
        // allows us to modify the switch at a later stage
        cond_.push_back_case(first_.value());
        // choose an ID which should be unique: to pick a very large number we cast a negative number to an unsigned
        // ID is unique
        typedef conditional< (uint_t) - (sizeof...(Cases)), Condition::index_value > conditional_t;

        uint_t rec_depth_ = 0;
        cond_.push_back_condition([&cond_, &first_]() { return (short_t)cond_.value()() == (short_t)first_.value(); });

        return if_(conditional_t((*cond_.m_conditions)[rec_depth_]),
            first_.mss(),
            recursive_switch(rec_depth_, cond_, cases_...));
    }

    template < typename Condition, typename First, typename... Cases >
    auto recursive_switch(uint_t recursion_depth_, Condition &cond_, First const &first_, Cases const &... cases_)
        -> decltype(if_(conditional< (uint_t) - (sizeof...(Cases)), Condition::index_value >(),
            first_.mss(),
            recursive_switch(recursion_depth_, cond_, cases_...))) {
        GRIDTOOLS_STATIC_ASSERT(
            (is_case_type< First >::value), "the entries in a switch_ statement must be case_ statements");

        GRIDTOOLS_STATIC_ASSERT((is_switch_variable< Condition >::value),
            "the first argument of the switch_ statement must be of switch_variable type");

        // save the boolean in a vector owned by the switch variable
        // allows us to modify the switch at a later stage
        cond_.push_back_case(first_.value());
        // choose an ID which should be unique: to pick a very large number we cast a negative number to an unsigned
        typedef conditional< (uint_t) - (sizeof...(Cases)), Condition::index_value > conditional_t;
        recursion_depth_++;

        cond_.push_back_condition([&cond_, &first_]() { return (short_t)cond_.value()() == (short_t)first_.value(); });

        return if_(conditional_t((*cond_.m_conditions)[recursion_depth_]),
            first_.mss(),
            recursive_switch(recursion_depth_, cond_, cases_...));
    }

    /**@brief recursion anchor*/
    template < typename Condition, typename Default >
    // typename switch_type<Condition, Default>::type
    typename Default::mss_t recursive_switch(
        uint_t /*recursion_depth_*/, Condition const & /*cond_*/, Default const &last_) {
        GRIDTOOLS_STATIC_ASSERT(
            (is_default_type< Default >::value), "the last entry in a switch_ statement must be a default_ statement");
        return last_.mss(); // default_ value
    }

} // namespace gridtools
