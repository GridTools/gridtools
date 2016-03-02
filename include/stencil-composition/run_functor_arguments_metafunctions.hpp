#pragma once

namespace gridtools {

    template < typename IterateDomainArguments >
    struct iterate_domain_arguments_reduction_type {
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<typename IterateDomainArguments::EsfSequence>::value == 1), "Error");
        GRIDTOOLS_STATIC_ASSERT((IterateDomainArguments::s_is_reduction), "Error");

        typedef typename boost::mpl::at<typename IterateDomainArguments::EsfSequence, static_uint<0> >::type esf_t;
        typedef typename esf_t::esf_function esf_function_t;
        typedef double type;
    };

} // gridtools
