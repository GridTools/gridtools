/*
  @file
  This file provides functionality for a iterate domain evaluator that intercepts calls to iterate domain
  and remap the arguments to the actual positions in the iterate domain
*/


#pragma once
#include "iterate_domain_metafunctions.h"
#include "accessor_metafunctions.h"

namespace gridtools {

namespace _impl {
    template<typename T>
    struct iterate_domain_evaluator_base_iterate_domain;

    template<
        typename IterateDomain, typename EsfArgsMap,
        template <typename, typename> class Impl>
    struct iterate_domain_evaluator_base_iterate_domain<Impl<IterateDomain, EsfArgsMap> >
    {
        typedef IterateDomain type;
    };

    template<typename T>
    struct iterate_domain_evaluator_base_esf_args_map;

    template<
        typename IterateDomain, typename EsfArgsMap,
        template <typename, typename> class Impl>
    struct iterate_domain_evaluator_base_esf_args_map<Impl<IterateDomain, EsfArgsMap> >
    {
        typedef EsfArgsMap type;
    };
}

/**
 * @class iterate_domain_evaluator_base
 * base class of an iterate_domain_evaluator that intercepts the calls to evaluate the value of an arguments
 * from the iterate domain, and redirect the arg specified by user to the actual position of the arg in the
 * iterate domain
 * @param IterateDomainEvaluatorImpl implementer class of the CRTP
 */
    template<typename IterateDomain, typename EsfArgsMap>
class iterate_domain_evaluator_base
{
DISALLOW_COPY_AND_ASSIGN(iterate_domain_evaluator_base);
public:
    typedef IterateDomain iterate_domain_t;
    typedef EsfArgsMap esf_args_map_t;

    BOOST_STATIC_ASSERT((is_iterate_domain<iterate_domain_t>::value));
    typedef typename iterate_domain_local_domain<iterate_domain_t>::type local_domain_t;

    GT_FUNCTION
    explicit iterate_domain_evaluator_base(const iterate_domain_t& iterate_domain) : m_iterate_domain(iterate_domain) {}

    template <typename AccessorType>
    GT_FUNCTION
    typename boost::mpl::at<
        typename local_domain_t::esf_args,
        typename AccessorType::type::index_type
    >::type::value_type& RESTRICT
    operator()(AccessorType const& accessor) const {
        typedef typename remap_accessor_type<AccessorType, esf_args_map_t>::type remap_accessor_t;
        return m_iterate_domain(remap_accessor_t(accessor));
    }

protected:
    const iterate_domain_t& m_iterate_domain;
};

/**
 * @class iterate_domain_evaluator
 * default iterate domain evaluator when positional information is not required
 * @param IterateDomain iterate domain
 * @param EsfArgsMap map from ESF arguments to iterate domain position of args.
 */
template<typename IterateDomain, typename EsfArgsMap>
class iterate_domain_evaluator : public
iterate_domain_evaluator_base<IterateDomain, EsfArgsMap>
{
DISALLOW_COPY_AND_ASSIGN(iterate_domain_evaluator);
public:
    BOOST_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value));
    typedef iterate_domain_evaluator_base<IterateDomain, EsfArgsMap > super;

    GT_FUNCTION
    explicit iterate_domain_evaluator(const IterateDomain& iterate_domain) : super(iterate_domain) {}

};

} // namespace gridtools
