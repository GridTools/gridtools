#pragma once


namespace gridtools
{
    template<typename IterateDomainArguments>
    struct reduced_data
    {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), "Internal error: wrong type");
        typename IterateDomainArguments::functor_return_type_t data;
    };

    namespace impl{
        template<typename IterateDomainArguments, bool IsReduction>
        class iterate_domain_reduction_impl{};

        template<typename IterateDomainArguments>
        class iterate_domain_reduction_impl<IterateDomainArguments, true>
        {
            typedef typename IterateDomainArguments::functor_return_type_t reduced_value_t;

        public:
            GT_FUNCTION
            reduced_value_t& reduced_value()
            {

            }
        private:
            reduced_value_t m_reduced_value;
        };

    }

    template<typename IterateDomainArguments>
    class iterate_domain_reduction :
            public impl::iterate_domain_reduction_impl<IterateDomainArguments, IterateDomainArguments::s_is_reduction>
    {};

}
