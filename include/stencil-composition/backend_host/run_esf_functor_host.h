/*
 * run_esf_functor_host.h
 *
 *  Created on: Apr 27, 2015
 *      Author: cosuna
 */
#pragma once
#include "../run_esf_functor.h"

namespace gridtools {
    template < typename RunFunctorArguments, typename Interval>
    struct run_esf_functor_host : public
        run_esf_functor<run_esf_functor_host<RunFunctorArguments, Interval> > //CRTP
    {
        typedef run_esf_functor<run_esf_functor_host<RunFunctorArguments, Interval> > super;
        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;

        GT_FUNCTION
        explicit run_esf_functor_host(iterate_domain_t& iterate_domain) : super(iterate_domain) {}

        template<typename IntervalType, typename EsfArguments>
        GT_FUNCTION
        void DoImpl() const
        {
            BOOST_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value));
            typedef typename EsfArguments::functor_t functor_t;
            functor_t::Do(this->m_iterate_domain, IntervalType());
        }

    };
}
