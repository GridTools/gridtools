#pragma once
#include "../run_esf_functor.hpp"

namespace gridtools {

    /*
     * @brief main functor that executes (for host) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    template < typename RunFunctorArguments, typename Interval>
    struct run_esf_functor_host : public
        run_esf_functor<run_esf_functor_host<RunFunctorArguments, Interval> > //CRTP
    {
        GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), "Internal Error: wrong type");
        typedef run_esf_functor<run_esf_functor_host<RunFunctorArguments, Interval> > super;
        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;

        GT_FUNCTION
        explicit run_esf_functor_host(iterate_domain_t& iterate_domain) : super(iterate_domain) {}

        /*
         * @brief main functor implemenation that executes (for Host) the user functor of an ESF
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template<typename IntervalType, typename EsfArguments>
        GT_FUNCTION
        void do_impl() const
        {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value), "Internal Error: wrong type");
            typedef typename EsfArguments::functor_t functor_t;
            functor_t::Do(this->m_iterate_domain, IntervalType());
        }

    };
}
