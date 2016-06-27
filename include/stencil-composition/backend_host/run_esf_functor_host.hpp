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
#include "../run_esf_functor.hpp"

namespace gridtools {

    /*
     * @brief main functor that executes (for host) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    template < typename RunFunctorArguments, typename Interval >
    struct run_esf_functor_host
        : public run_esf_functor< run_esf_functor_host< RunFunctorArguments, Interval > > // CRTP
    {

        GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), "Internal Error: wrong type");
        typedef run_esf_functor< run_esf_functor_host< RunFunctorArguments, Interval > > super;
        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;

        GT_FUNCTION
        explicit run_esf_functor_host(iterate_domain_t &iterate_domain) : super(iterate_domain) {}

        /*
         * @brief main functor implemenation that executes (for Host) the user functor of an ESF
         *      (specialization for non reduction operations)
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template < typename IntervalType, typename EsfArguments >
        GT_FUNCTION void do_impl(
            typename boost::disable_if< typename EsfArguments::is_reduction_t, int >::type = 0) const {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), "Internal Error: wrong type");
            typedef typename EsfArguments::functor_t functor_t;

            // GRIDTOOLS_STATIC_ASSERT(functor_t::repeat_t::value>0, "internal error");
            _impl::call_repeated< functor_t::repeat_t::value, functor_t, iterate_domain_t, IntervalType >::Do(
                this->m_iterate_domain);
        }

        /*
         * @brief main functor implemenation that executes (for Host) the user functor of an ESF
         *      (specialization for reduction operations)
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template < typename IntervalType, typename EsfArguments >
        GT_FUNCTION void do_impl(
            typename boost::enable_if< typename EsfArguments::is_reduction_t, int >::type = 0) const {
            typedef typename EsfArguments::reduction_data_t::bin_op_t bin_op_t;
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), "Internal Error: wrong type");
            typedef typename EsfArguments::functor_t functor_t;
            this->m_iterate_domain.set_reduction_value(bin_op_t()(this->m_iterate_domain.reduction_value(),
                functor_t::f_type::Do(this->m_iterate_domain, IntervalType())));
        }
    };
}
