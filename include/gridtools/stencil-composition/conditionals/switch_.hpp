/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <functional>
#include <type_traits>

#include "../../common/defs.hpp"
#include "./case_type.hpp"
#include "./if_.hpp"
/**@file
 */

namespace gridtools {
    /**@brief API for a runtime switch between several multi stage stencils

       Its implementation is recursive. It creates as many boolean conditionals as
       the number of \ref gridtools::case_ defined, and it implements the switch in terms of
       \ref gridtools::if_ constructs. The unique ID which is necessary in order to define the boolean
       conditionals in this case is assigned automatically by the library (as a very large number)

       \tparam Cond must be a nullary integer functor.
       \tparam First must be an instance of @ref gridtools::case_type
       \tparam Cases must be instances of @ref gridtools::case_type, or @ref gridtools::default_type fir the last one

       \param cond_ an instance of type Cond
       \param first_ the first case
       \param cases_ the pack of cases, the last one being the default_ one

       NOTE: multiple switch statements can coexist in the same computation,
       and the arbitrary nesting of switch_ statements are supported.


       example of usage:
@verbatim
auto c1 = []{ return 3; }

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

    template <typename Fun, typename Val>
    struct case_adapter {
        Fun m_fun;
        Val m_val;
        bool operator()() const { return m_fun() == m_val; }
    };
    template <typename Fun, typename Val>
    case_adapter<Fun, Val> make_case_adapter(Fun const &fun, Val val) {
        return {fun, val};
    }

    /**@brief recursion anchor*/
    template <typename Cond, typename Default>
    typename Default::mss_t switch_(Cond const & /*cond_*/, Default const &last_) {
        GT_STATIC_ASSERT(
            (is_default_type<Default>::value), "the last entry in a switch_ statement must be a default_ statement");
        return last_.mss(); // default_ value
    }

    template <typename Cond, typename First, typename... Cases>
    auto switch_(Cond const &cond_, First const &first_, Cases const &... cases_)
        -> decltype(if_(make_case_adapter(cond_, first_.value()), first_.mss(), switch_(cond_, cases_...))) {
        GT_STATIC_ASSERT((std::is_convertible<Cond, std::function<int()>>::value),
            "switch_ argument should be a nullary integer functor");
        GT_STATIC_ASSERT((is_case_type<First>::value), "the entries in a switch_ statement must be case_ statements");
        return if_(make_case_adapter(cond_, first_.value()), first_.mss(), switch_(cond_, cases_...));
    }
} // namespace gridtools
