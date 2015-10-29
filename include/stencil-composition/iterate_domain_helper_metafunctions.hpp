#pragma once

namespace gridtools
{
    /**
     * metafunction that retrieves the arg type associated with an accessor
     */
    template<typename Accessor, typename IterateDomainArguments>
    struct get_arg_from_accessor
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), "Internal error: wrong type");

        typedef typename IterateDomainArguments::local_domain_t local_domain_t;
        typedef typename local_domain_t::esf_args esf_args_t;

        typedef typename boost::mpl::at<
            esf_args_t,
            typename Accessor::index_type
        >::type type;
    };

    /**
     * metafunction that computes the return type of all operator() of an accessor
     */
    template<typename Accessor, typename IterateDomainArguments>
    struct accessor_return_type
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), "Internal error: wrong type");

        typedef typename boost::mpl::eval_if<
            is_accessor<Accessor>,
            get_arg_from_accessor<Accessor, IterateDomainArguments>,
            boost::mpl::identity<boost::mpl::void_>
        >::type type;
    };

} //namespace
