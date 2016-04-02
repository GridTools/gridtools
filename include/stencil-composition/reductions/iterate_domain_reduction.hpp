#pragma once

namespace gridtools {

    namespace impl {
        template < typename IterateDomainArguments, bool IsReduction >
        class iterate_domain_reduction_impl {
            typedef typename IterateDomainArguments::functor_return_type_t reduction_type_t;

          public:
            GT_FUNCTION
            iterate_domain_reduction_impl(const reduction_type_t &initial_value) {}
            GT_FUNCTION
            reduction_type_t reduction_value() const { return 0; }
        };

        template < typename IterateDomainArguments >
        class iterate_domain_reduction_impl< IterateDomainArguments, true > {
          protected:
            typedef typename IterateDomainArguments::functor_return_type_t reduction_type_t;

          public:
            GT_FUNCTION
            iterate_domain_reduction_impl(const reduction_type_t &initial_value) : m_reduced_value(initial_value) {}

            GT_FUNCTION
            reduction_type_t reduction_value() const { return m_reduced_value; }

            GT_FUNCTION
            void set_reduction_value(reduction_type_t value) { m_reduced_value = value; }

          private:
            reduction_type_t m_reduced_value;
        };
    }

    template < typename IterateDomainArguments >
    class iterate_domain_reduction
        : public impl::iterate_domain_reduction_impl< IterateDomainArguments, IterateDomainArguments::s_is_reduction > {
      public:
        typedef typename IterateDomainArguments::functor_return_type_t reduction_type_t;

        GT_FUNCTION
        iterate_domain_reduction(const reduction_type_t &initial_value)
            : impl::iterate_domain_reduction_impl< IterateDomainArguments, IterateDomainArguments::s_is_reduction >(
                  initial_value) {}
    };
}
