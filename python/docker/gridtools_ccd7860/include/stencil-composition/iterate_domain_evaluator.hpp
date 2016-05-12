/*
  @file
  This file provides functionality for a iterate domain evaluator that intercepts calls to iterate domain
  and remap the arguments to the actual positions in the iterate domain
*/


#pragma once
#include "iterate_domain_metafunctions.hpp"
#include "accessor_metafunctions.hpp"

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
template<typename IterateDomainEvaluatorImpl>
class iterate_domain_evaluator_base
{
DISALLOW_COPY_AND_ASSIGN(iterate_domain_evaluator_base);

    typedef typename _impl::iterate_domain_evaluator_base_iterate_domain<IterateDomainEvaluatorImpl>::type iterate_domain_t;
protected:
    const iterate_domain_t& m_iterate_domain;

public:

    typedef typename _impl::iterate_domain_evaluator_base_esf_args_map<IterateDomainEvaluatorImpl>::type esf_args_map_t;

    GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<iterate_domain_t>::value), "Internal Error: wrong type");
    typedef typename iterate_domain_local_domain<iterate_domain_t>::type local_domain_t;


    GT_FUNCTION
    explicit iterate_domain_evaluator_base(const iterate_domain_t& iterate_domain) : m_iterate_domain(iterate_domain) {}


    /** shifting the IDs of the placeholders and forwarding to the iterate_domain () operator*/
    template <typename Expression>
    GT_FUNCTION
#ifdef CXX11_ENABLED
    auto
    operator() (Expression const&  arg) const -> decltype(m_iterate_domain(arg))
#else
    typename boost::mpl::at<
        typename local_domain_t::esf_args,
        typename Expression::type::index_type
    >::type::value_type& RESTRICT
    operator() (Expression const&  arg) const
#endif
    {
        typedef typename remap_accessor_type<Expression, esf_args_map_t>::type remap_accessor_t;
        return m_iterate_domain(remap_accessor_t(arg));
    }

protected:
};

/**
 * @class iterate_domain_evaluator
 * default iterate domain evaluator when positional information is not required
 * @param IterateDomain iterate domain
 * @param EsfArgsMap map from ESF arguments to iterate domain position of args.
 */
template<typename IterateDomain, typename EsfArgsMap>
class iterate_domain_evaluator : public
    iterate_domain_evaluator_base<iterate_domain_evaluator<IterateDomain, EsfArgsMap> > //CRTP
{
DISALLOW_COPY_AND_ASSIGN(iterate_domain_evaluator);
public:
    GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), "Internal Error: wrong type");
    typedef iterate_domain_evaluator_base<iterate_domain_evaluator<IterateDomain, EsfArgsMap> > super;

    GT_FUNCTION
    explicit iterate_domain_evaluator(const IterateDomain& iterate_domain) : super(iterate_domain) {}

};

/**
 * @class positional_iterate_domain_evaluator
 * iterate domain evaluator when positional information is required
 * @param IterateDomain iterate domain
 * @param EsfArgsMap map from ESF arguments to iterate domain position of args.
 */
template<typename IterateDomain, typename EsfArgsMap>
class positional_iterate_domain_evaluator : public
    iterate_domain_evaluator_base<positional_iterate_domain_evaluator<IterateDomain, EsfArgsMap> > //CRTP
{
DISALLOW_COPY_AND_ASSIGN(positional_iterate_domain_evaluator);
public:
    GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), "Internal Error: wrong type");
    typedef iterate_domain_evaluator_base<positional_iterate_domain_evaluator<IterateDomain, EsfArgsMap> > super;

    GT_FUNCTION
    explicit positional_iterate_domain_evaluator(const IterateDomain& iterate_domain) : super(iterate_domain) {}

    GT_FUNCTION
    uint_t i() const {
        return this->m_iterate_domain.i();
    }

    GT_FUNCTION
    uint_t j() const {
        return this->m_iterate_domain.j();
    }

    GT_FUNCTION
    uint_t k() const {
        return this->m_iterate_domain.k();
    }
};

/**
 * @struct get_iterate_domain_evaluator
 * metafunction that computes the iterate_domain_evaluator from the iterate domain type
 */
template<typename IterateDomain, typename EsfArgsMap>
struct get_iterate_domain_evaluator
{
    GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), "Internal Error: wrong type");
    template<typename _IterateDomain, typename _EsfArgsMap>
    struct select_basic_iterate_domain_evaluator
    {
        typedef iterate_domain_evaluator<_IterateDomain, _EsfArgsMap> type;
    };
    template<typename _IterateDomain, typename _EsfArgsMap>
    struct select_positional_iterate_domain_evaluator
    {
        typedef positional_iterate_domain_evaluator<_IterateDomain, _EsfArgsMap> type;
    };

    typedef typename boost::mpl::eval_if<
        is_positional_iterate_domain<IterateDomain>,
        select_positional_iterate_domain_evaluator<IterateDomain, EsfArgsMap>,
        select_basic_iterate_domain_evaluator<IterateDomain, EsfArgsMap>
    >::type type;

};

} // namespace gridtools
