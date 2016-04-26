#pragma once
#include "../run_esf_functor.hpp"
#include "../expandable_parameters/iterate_domain_expandable_parameters.hpp"

namespace gridtools {

    namespace _impl{
    template<ushort_t ID, typename Functor, typename IterateDomain, typename Interval>
    struct call_repeated{
    public:

        static void Do(IterateDomain& it_domain_){

            Functor::f_type::Do(
                *static_cast<iterate_domain_expandable_parameters<
                IterateDomain
                , ID> *
                > (&it_domain_), Interval());

            call_repeated<ID-1, Functor, IterateDomain, Interval>::Do(it_domain_);
        }
    };

    template<typename Functor, typename IterateDomain, typename Interval>
    struct call_repeated<0, Functor, IterateDomain, Interval>{
    public:
        static void Do(IterateDomain& it_domain_){
        }
    };
    }//namespace _impl

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
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template < typename IntervalType, typename EsfArguments >
        GT_FUNCTION void do_impl() const {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), "Internal Error: wrong type");
            typedef typename EsfArguments::functor_t functor_t;

            // GRIDTOOLS_STATIC_ASSERT(functor_t::repeat_t::value>0, "internal error");
            _impl::call_repeated<functor_t::repeat_t::value, functor_t, iterate_domain_t, IntervalType>::Do(this->m_iterate_domain);
        }
    };
}
