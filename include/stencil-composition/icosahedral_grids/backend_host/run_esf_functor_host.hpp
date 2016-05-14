#pragma once

#include "../../run_esf_functor.hpp"

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

        template < typename EsfArguments >
        struct color_esf_match {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), "Error");
            typedef typename boost::mpl::or_< typename boost::is_same< typename RunFunctorArguments::color_t,
                                                  typename EsfArguments::esf_t::color_t >::type,
                typename boost::is_same< notype, typename EsfArguments::esf_t::color_t >::type >::type type;
        };

        /*
         * @brief main functor implemenation that executes (for Host) the user functor of an ESF
         *      (specialization for non reduction operations)
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template < typename IntervalType, typename EsfArguments >
        GT_FUNCTION void do_impl(
            typename boost::enable_if< typename color_esf_match< EsfArguments >::type, int >::type = 0) const {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), "Internal Error: wrong type");

            call_user_functor<IntervalType, EsfArguments>();
        }

        template < typename IntervalType, typename EsfArguments >
        GT_FUNCTION void do_impl(
            typename boost::disable_if< typename color_esf_match< EsfArguments >::type, int >::type = 0) const {}

    private:
        /*
         * @brief main functor implemenation that executes (for Host) the user functor of an ESF
         *      (specialization for non reduction operations)
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template < typename IntervalType, typename EsfArguments >
        GT_FUNCTION void call_user_functor(
            typename boost::disable_if< typename EsfArguments::is_reduction_t, int >::type = 0) const {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), "Internal Error: wrong type");
            typedef typename EsfArguments::functor_t functor_t;

            using n_colors_t = typename EsfArguments::esf_t::location_type::n_colors;

            functor_t::f_type::Do(this->m_iterate_domain, IntervalType());
        }

        /*
         * @brief main functor implemenation that executes (for Host) the user functor of an ESF
         *      (specialization for reduction operations)
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template < typename IntervalType, typename EsfArguments >
        GT_FUNCTION void call_user_functor(
            typename boost::enable_if< typename EsfArguments::is_reduction_t, int >::type = 0) const {
            typedef typename EsfArguments::reduction_data_t::bin_op_t bin_op_t;
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), "Internal Error: wrong type");
            typedef typename EsfArguments::functor_t functor_t;
            this->m_iterate_domain.set_reduction_value(bin_op_t()(this->m_iterate_domain.reduction_value(),
                functor_t::f_type::Do(this->m_iterate_domain, IntervalType())));
        }
    };
}
