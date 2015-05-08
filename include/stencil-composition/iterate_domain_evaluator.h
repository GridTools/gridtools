/*
  @file
  This file provides functionality for a iterate domain evaluator that intercepts calls to iterate domain
  and remap the arguments to the actual positions in the iterate domain
*/


#pragma once
#include "iterate_domain_metafunctions.h"

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
 * @brief metafunction that given an arg and a map, it will remap the index of the arg according
 * to the corresponding entry in ArgsMap
 */
template<typename Arg, typename ArgsMap>
struct remap_arg_type;

//template<uint_t I, typename Range, ushort_t Dim, typename ArgsMap >
//struct remap_arg_type<arg_type_base<I, Range, Dim>, ArgsMap >
//{
//    typedef arg_type_base<I, Range, Dim> base_arg_t;
//    BOOST_STATIC_ASSERT((boost::mpl::size<ArgsMap>::value>0));
//    //check that the key type is an int (otherwise the later has_key would never find the key)
//    BOOST_STATIC_ASSERT((boost::is_same<
//        typename boost::mpl::first<typename boost::mpl::front<ArgsMap>::type>::type::value_type,
//        int
//    >::value));
//
//    typedef typename boost::mpl::integral_c<int, base_arg_t::index_type::value> index_type_t;
//
//    BOOST_STATIC_ASSERT((boost::mpl::has_key<ArgsMap, index_type_t>::value));
//
//    typedef arg_type_base<
//        boost::mpl::at<ArgsMap, index_type_t >::type::value,
//        typename base_arg_t::range_type,
//        base_arg_t::n_dim
//    > type;
//};

template<ushort_t I, typename Range, ushort_t Dim, typename ArgsMap >
struct remap_arg_type<arg_type<I, Range, Dim>, ArgsMap >
{
    typedef arg_type<I, Range, Dim> base_arg_t;
    BOOST_STATIC_ASSERT((boost::mpl::size<ArgsMap>::value>0));
    //check that the key type is an int (otherwise the later has_key would never find the key)
    BOOST_STATIC_ASSERT((boost::is_same<
        typename boost::mpl::first<typename boost::mpl::front<ArgsMap>::type>::type::value_type,
        int
    >::value));

    typedef typename boost::mpl::integral_c<int, base_arg_t::index_type::value> index_type_t;

    BOOST_STATIC_ASSERT((boost::mpl::has_key<ArgsMap, index_type_t>::value));

    typedef arg_type<
        boost::mpl::at<ArgsMap, index_type_t >::type::value,
        typename base_arg_t::range_type,
        base_arg_t::n_dim
    > type;
};


template< class ArgType, typename ArgsMap >
struct remap_arg_type<arg_decorator<ArgType>, ArgsMap >
{
    typedef arg_decorator<typename remap_arg_type<ArgType, ArgsMap>::type > type;
};

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
public:
    typedef typename _impl::iterate_domain_evaluator_base_iterate_domain<IterateDomainEvaluatorImpl>::type iterate_domain_t;
    typedef typename _impl::iterate_domain_evaluator_base_esf_args_map<IterateDomainEvaluatorImpl>::type esf_args_map_t;

    BOOST_STATIC_ASSERT((is_iterate_domain<iterate_domain_t>::value));
    typedef typename iterate_domain_local_domain<iterate_domain_t>::type local_domain_t;

    GT_FUNCTION
    explicit iterate_domain_evaluator_base(const iterate_domain_t& iterate_domain) : m_iterate_domain(iterate_domain) {}

    template <typename ArgType>
    GT_FUNCTION
    typename boost::enable_if<
        typename boost::mpl::bool_< (ArgType::type::n_args <=
                                     boost::mpl::at<
                                     typename local_domain_t::esf_args
                                     , typename ArgType::type::index_type>::type::storage_type::space_dimensions)>::type
                                    , typename boost::mpl::at<typename local_domain_t::esf_args
                                                              , typename ArgType::type::index_type>::type::value_type
                                    >::type& RESTRICT
    operator()(ArgType const& arg) const {
        typedef typename remap_arg_type<ArgType, esf_args_map_t>::type remap_arg_t;
        return m_iterate_domain(remap_arg_t(arg));
    }

    template < typename ArgType>
    GT_FUNCTION
    typename boost::enable_if<
        typename boost::mpl::bool_<(ArgType::type::n_args >
        boost::mpl::at<
        typename local_domain_t::esf_args
        , typename ArgType::type::index_type>::type::storage_type::space_dimensions)>::type
        , typename boost::mpl::at<typename local_domain_t::esf_args
        , typename ArgType::type::index_type>::type::value_type
    >::type&  RESTRICT
    operator()(ArgType const& arg) const {
        typedef typename remap_arg_type<ArgType, esf_args_map_t>::type remap_arg_t;
        return m_iterate_domain(remap_arg_t(arg));
    }

#ifdef CXX11_ENABLED
    /** @brief method called in the Do methods of the functors.
        specialization for the expr_direct_access<arg_type> placeholders
    */
    template <typename ArgType>
    GT_FUNCTION
    typename boost::mpl::at<typename local_domain_t::esf_args, typename arg_decorator<ArgType>::index_type>::type::value_type&
    operator()(expr_direct_access<arg_decorator<ArgType> > const& arg) const {
        typedef typename remap_arg_type<ArgType, esf_args_map_t>::type remap_arg_t;
        return m_iterate_domain(remap_arg_t(arg));
    }

    /** @brief method called in the Do methods of the functors.
        Specialization for the arg_decorator placeholder (i.e. for extended storages, containg multiple snapshots of data fields with the same dimension and memory layout)*/
    template < typename ArgType>
    GT_FUNCTION
    typename boost::enable_if<
        typename boost::mpl::bool_<(gridtools::arg_decorator<ArgType>::n_args > boost::mpl::at<typename local_domain_t::esf_args, typename ArgType::index_type>::type::storage_type::space_dimensions)>::type,
        typename boost::mpl::at<typename local_domain_t::esf_args, typename ArgType::index_type>::type::value_type& >::type
    operator()(gridtools::arg_decorator<ArgType> const&& arg) const {
        return m_iterate_domain(remap_arg_t(std::forward<gridtools::arg_decorator<ArgType> const> arg));
    }

#if !defined(__CUDACC__)
        /** @brief method called in the Do methods of the functors.

            Specialization for the arg_decorator placeholder (i.e. for extended storages, containg multiple snapshots of data fields with the same dimension and memory layout)*/
        template < typename ArgType, typename ... Pairs>
        GT_FUNCTION
        typename boost::mpl::at<typename local_domain_t::esf_args
                                , typename ArgType::index_type>::type::value_type& RESTRICT
        operator()(arg_mixed<ArgType, Pairs ... > const& arg) const
        {
//TODOCOSUNA implement
        }
        template <typename ... Arguments, template<typename ... Args> class Expression >
        GT_FUNCTION
        auto operator() (Expression<Arguments ... > const& arg) const ->decltype(evaluation::value(*this, arg)) {
            //arg.to_string();
            return evaluation::value((*this), arg);
        }

        /** @brief method called in the Do methods of the functors.
            partial specializations for double (or float)*/
        template <typename Arg, template<typename Arg1, typename Arg2> class Expression, typename FloatType
                  , typename boost::enable_if<typename boost::is_floating_point<FloatType>::type, int >::type=0 >
        GT_FUNCTION
        auto operator() (Expression<Arg, FloatType> const& arg) const ->decltype(evaluation::value_scalar(*this, arg)) {
        }

        /** @brief method called in the Do methods of the functors.
            partial specializations for int. Here we do not use the typedef int_t, because otherwise the interface would be polluted with casting
            (the user would have to cast all the numbers (-1, 0, 1, 2 .... ) to int_t before using them in the expression)*/
        // template <typename Arg, int Arg2, template<typename Arg1, int a> class Expression >
        template <typename Arg, template<typename Arg1, typename Arg2> class Expression, typename IntType
                  , typename boost::enable_if<typename boost::is_integral<IntType>::type, int >::type=0 >
        GT_FUNCTION
        auto operator() (Expression<Arg, IntType> const& arg) const ->decltype(evaluation::value_int((*this), arg)) {
        }

        template <typename Arg, template<typename Arg1, int Arg2> class Expression
                  , /*typename IntType, typename boost::enable_if<typename boost::is_integral<IntType>::type, int >::type=0*/int exponent >
        GT_FUNCTION
        auto operator() (Expression<Arg, exponent> const& arg) const ->decltype(evaluation::value_int((*this), arg)) {
        }
#endif

#endif
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
    iterate_domain_evaluator_base<iterate_domain_evaluator<IterateDomain, EsfArgsMap> > //CRTP
{
DISALLOW_COPY_AND_ASSIGN(iterate_domain_evaluator);
public:
    BOOST_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value));
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
    BOOST_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value));
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
 * metafunctions that computes the iterate_domain_evaluator from the iterate domain type
 */
template<typename IterateDomain, typename EsfArgsMap>
struct get_iterate_domain_evaluator
{
    BOOST_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value));
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
