#pragma once

#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include "../accessor.hpp"
#include "call_interfaces_metafunctions.hpp"
#include "../../common/generic_metafunctions/v_item_to_fusion_vector.hpp"

namespace gridtools {



    namespace _impl {
        /**
           In the context of stencil_functions, this type represents the
           aggregator/domain/evaluator to be passed to a stencil function,
           called within a stencil operator or another stencil function.
           Construct a new aggregator/domain from the initial one (which is
           usually an iterate_domain).  The indices in the seauence
           UsedIndices indicate which elements of the CallerAggregator are
           passed to the function. One of the indices correspond to the output
           argument which is a scalar and requires special attention.
        */
        template <typename CallerAggregator, int Offi, int Offj, int Offk,
                  typename PassedAccessors, typename ReturnType, int OutArg>
        struct function_aggregator {
            CallerAggregator const& m_caller_aggregator;
            ReturnType __restrict__ * m_result;

            GT_FUNCTION
            function_aggregator(CallerAggregator const& caller_aggregator, ReturnType & result)
                : m_caller_aggregator(caller_aggregator)
                , m_result(&result)
            {}

            template <typename Accessor>
            GT_FUNCTION
            constexpr
            typename boost::enable_if_c<(Accessor::index_type::value < OutArg), ReturnType>::type const
            operator()(Accessor const& accessor) const {
                return m_caller_aggregator
                    (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value>::type
                     (accessor.template get<2>()+Offi,
                      accessor.template get<1>()+Offj,
                      accessor.template get<0>()+Offk));
            }

                template <typename Accessor>
                GT_FUNCTION
                constexpr
                typename boost::enable_if_c<(Accessor::index_type::value > OutArg), ReturnType>::type const
                operator()(Accessor const& accessor) const {
                    return m_caller_aggregator
                        (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value-1>::type
                         (accessor.template get<2>()+Offi,
                          accessor.template get<1>()+Offj,
                          accessor.template get<0>()+Offk));
                }

            template <typename Accessor>
            GT_FUNCTION
            constexpr
            typename boost::enable_if_c<(Accessor::index_type::value == OutArg), ReturnType>::type&
            operator()(Accessor const&) const {
                // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
                return *m_result;
            }

            /** @brief method called in the Do methods of the functors. */
            template <typename ... Arguments, template<typename ... Args> class Expression >
            GT_FUNCTION
            constexpr
            auto operator() (Expression<Arguments ... > const& arg) const
                ->decltype(evaluation::value(*this, arg))
            {
                //arg.to_string();
                return evaluation::value((*this), arg);
            }

            /** @brief method called in the Do methods of the functors.
                partial specializations for double (or float)*/
            template <typename Accessor, template<typename Arg1, typename Arg2> class Expression, typename FloatType
                      , typename boost::enable_if<typename boost::is_floating_point<FloatType>::type, int >::type=0 >
            GT_FUNCTION
            constexpr
            auto operator() (Expression<Accessor, FloatType> const& arg) const
                ->decltype(evaluation::value_scalar(*this, arg)) {
                //TODO RENAME ACCESSOR,is not an accessor but an expression, and add an assertion for type
                return evaluation::value_scalar((*this), arg);
            }
        };


        /**
           In the context of stencil_functions, this type represents the
           aggregator/domain/evaluator to be passed to a stencil function,
           called within a stencil operator or another stencil function.
           Construct a new aggregator/domain from the initial one (which is
           usually an iterate_domain).  The indices in the seauence
           UsedIndices indicate which elements of the CallerAggregator are
           passed to the function. One of the indices correspond to the output
           argument which is a scalar and requires special attention.
        */
        template <typename CallerAggregator, int Offi, int Offj, int Offk,
                  typename PassedAccessors, typename ReturnType, int OutArg>
        struct function_aggregator_offsets {
            typedef typename boost::fusion::result_of::as_vector<PassedAccessors>::type accessors_list_t;
            CallerAggregator const& m_caller_aggregator;
            ReturnType __restrict__ * m_result;
            accessors_list_t const& m_accessors_list;

            GT_FUNCTION
            constexpr
            function_aggregator_offsets(CallerAggregator const& caller_aggregator,
                                        ReturnType & result,
                                        accessors_list_t const& list)
                : m_caller_aggregator(caller_aggregator)
                , m_result(&result)
                , m_accessors_list(list)
            {}

            template <typename Accessor>
            GT_FUNCTION
            constexpr
            typename boost::enable_if_c<(Accessor::index_type::value < OutArg), ReturnType>::type const
            operator()(Accessor const& accessor) const {
                return m_caller_aggregator
                    (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value>::type
                     (accessor.template get<2>()
                      +Offi
                      +boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<2>(),
                      accessor.template get<1>()
                      +Offj
                      +boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<1>(),
                      accessor.template get<0>()
                      +Offk
                      +boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<0>()));
            }

                template <typename Accessor>
                GT_FUNCTION
                constexpr
                typename boost::enable_if_c<(Accessor::index_type::value > OutArg), ReturnType>::type const
                operator()(Accessor const& accessor) const {
                    return m_caller_aggregator
                        (typename boost::mpl::at_c<PassedAccessors, Accessor::index_type::value-1>::type
                         (accessor.template get<2>()
                          +Offi
                          +boost::fusion::at_c<Accessor::index_type::value-1>(m_accessors_list).template get<2>(),
                          accessor.template get<1>()
                          +Offj
                          +boost::fusion::at_c<Accessor::index_type::value-1>(m_accessors_list).template get<1>(),
                          accessor.template get<0>()
                          +Offk
                          +boost::fusion::at_c<Accessor::index_type::value-1>(m_accessors_list).template get<0>()));
                }

            template <typename Accessor>
            GT_FUNCTION
            constexpr
            typename boost::enable_if_c<(Accessor::index_type::value == OutArg), ReturnType>::type&
            operator()(Accessor const&) const {
                // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
                return *m_result;
            }

        };
    } // namespace _impl


    /** Main interface for calling stencil operators as functions.

        Usage C++11: call<functor, region>::[at<offseti, offsetj, offsetk>::]with(eval, accessors...);

        Usage : call<functor, region>::[at<offseti, offsetj, offsetk>::type::]with(eval, accessors...);
    */
    template <typename Functor, typename Region, int Offi=0, int Offj=0, int Offk=0>
    struct call {
        template <int I, int J, int K>
        struct at_ {
            typedef call<Functor, Region, I, J, K> type;
        };

        template <int I, int J, int K>
        using at = call<Functor, Region, I, J, K>;

        template <typename Eval, typename Funct>
        struct get_result_type {
            typedef accessor<_impl::_get_index_of_first_non_const<Funct>::value> accessor_t;

            typedef typename Eval::template accessor_return_type
            <accessor_t>::type r_type;

            typedef typename std::decay<r_type>::type type;
        };

        template <typename Evaluator, typename ...Args>
        GT_FUNCTION
        static
        typename get_result_type<Evaluator, Functor>::type
        with_offsets(Evaluator const& eval, Args const& ...args) {

            static_assert(_impl::can_be_a_function<Functor>::value,
                          "Trying to invoke stencil operator with more than one output as a function\n");

            typedef typename get_result_type<Evaluator, Functor>::type result_type;
            typedef _impl::function_aggregator_offsets<
                Evaluator,
                Offi, Offj, Offk,
                typename gridtools::variadic_to_vector<Args...>::type,
                result_type,
                _impl::_get_index_of_first_non_const<Functor>::value> f_aggregator_t;

            result_type result;

            Functor::Do
                (
                 f_aggregator_t
                 (
                  eval,
                  result,
                  typename f_aggregator_t::accessors_list_t(args...)
                  ),
                 Region()
                 );

            return result;

        }

        template <typename Evaluator, typename ...Args>
        GT_FUNCTION
        static
        typename get_result_type<Evaluator, Functor>::type
        with(Evaluator const& eval, Args const & ...) {

            static_assert(_impl::can_be_a_function<Functor>::value,
                          "Trying to invoke stencil operator with more than one output as a function\n");

            typedef typename get_result_type<Evaluator, Functor>::type result_type;

            result_type result;
            typedef _impl::function_aggregator<
                Evaluator,
                Offi, Offj, Offk,
                typename gridtools::variadic_to_vector<Args...>::type,
                result_type,
                _impl::_get_index_of_first_non_const<Functor>::value> f_aggregator_t;

            Functor::Do(f_aggregator_t(eval, result), Region());

            return result;
        }
    };

    namespace _impl{
        template <typename CallerAggregator,
                  int Offi, int Offj, int Offk,
                  typename PassedArguments>
        struct function_aggregator_procedure_offsets {

            // Collect the indices of the arguments that are not accessors among
            // the PassedArguments
            typedef typename boost::mpl::fold<
                boost::mpl::range_c<int, 0, boost::mpl::size<PassedArguments>::value>,
                boost::mpl::vector0<>,
                typename _impl::insert_index_if_not_accessor<PassedArguments>:: template apply<boost::mpl::_2, boost::mpl::_1>
                >::type non_accessor_indices;

            //        typedef typename wrap_reference<PassedArguments>::type wrapped_accessors
            //typedef typename boost::fusion::result_of::as_vector<wrapped_accessors>::type accessors_list_t;
            typedef typename boost::fusion::result_of::as_vector<
                typename v_item_to_fusion_vector<PassedArguments>::type>::type accessors_list_t;

            CallerAggregator const& m_caller_aggregator;
            accessors_list_t const& m_accessors_list;

            GT_FUNCTION
            constexpr
            function_aggregator_procedure_offsets(CallerAggregator const& caller_aggregator,
                                                  accessors_list_t const & list)
                : m_caller_aggregator(caller_aggregator)
                , m_accessors_list(list)
            { }

            template <typename Accessor>
            GT_FUNCTION
            constexpr
            typename boost::lazy_enable_if_c<
                not _impl::contains_value<non_accessor_indices,
                                          typename Accessor::index_type>::value,
                typename CallerAggregator::template accessor_return_type<typename boost::mpl::at_c<PassedArguments, Accessor::index_type::value>::type>//::type
                >::type
            operator()(Accessor const& accessor) const {
                return m_caller_aggregator
                    (typename boost::mpl::at_c<PassedArguments, Accessor::index_type::value>::type
                     (accessor.template get<2>()
                      +Offi
                      +boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<2>(),
                      accessor.template get<1>()
                      +Offj
                      +boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<1>(),
                      accessor.template get<0>()
                      +Offk
                      +boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).template get<0>()));
            }

            template <typename Accessor>
            GT_FUNCTION
            constexpr
            typename boost::lazy_enable_if_c<
                _impl::contains_value<non_accessor_indices,
                                      typename Accessor::index_type>::value,
                typename boost::remove_reference<typename boost::fusion::result_of::at_c<accessors_list_t, Accessor::index_type::value>::type>::type//::type
                >::type&
            operator()(Accessor const&) const {
                // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
                return (boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).value());
            }
        };

        template <typename CallerAggregator,
                  int Offi, int Offj, int Offk,
                  typename PassedArguments>
        struct function_aggregator_procedure {

            // Collect the indices of the arguments that are not accessors among
            // the PassedArguments
            typedef typename boost::mpl::fold<
                boost::mpl::range_c<int, 0, boost::mpl::size<PassedArguments>::value>,
                boost::mpl::vector0<>,
                typename _impl::insert_index_if_not_accessor<PassedArguments>:: template apply<boost::mpl::_2, boost::mpl::_1>
                >::type non_accessor_indices;

            typedef typename boost::fusion::result_of::as_vector<
                typename v_item_to_fusion_vector<PassedArguments>::type>::type accessors_list_t;

            CallerAggregator const& m_caller_aggregator;
            accessors_list_t const& m_accessors_list;

            GT_FUNCTION
            function_aggregator_procedure(CallerAggregator const& caller_aggregator,
                                          accessors_list_t const & list)
                : m_caller_aggregator(caller_aggregator)
                , m_accessors_list(list)
            { }

            template <typename Accessor>
            GT_FUNCTION
            constexpr
            typename boost::lazy_enable_if_c<
                not _impl::contains_value<non_accessor_indices,
                                          typename Accessor::index_type>::value,
                typename CallerAggregator::template accessor_return_type<typename boost::mpl::at_c<PassedArguments, Accessor::index_type::value>::type>//::type
                >::type
            operator()(Accessor const& accessor) const {
                return m_caller_aggregator
                    (typename boost::mpl::at_c<PassedArguments, Accessor::index_type::value>::type
                     (accessor.template get<2>()+Offi,
                      accessor.template get<1>()+Offj,
                      accessor.template get<0>()+Offk));
            }

            template <typename Accessor>
            GT_FUNCTION
            constexpr
            typename boost::lazy_enable_if_c<
                _impl::contains_value<non_accessor_indices,
                                      typename Accessor::index_type>::value,
                typename boost::remove_reference<typename boost::fusion::result_of::at_c<accessors_list_t, Accessor::index_type::value>::type>::type//::type
                >::type&
            operator()(Accessor const&) const {
                // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
                return (boost::fusion::at_c<Accessor::index_type::value>(m_accessors_list).value());
            }
        };

        } //namespace _impl


    /** Main interface for calling stencil operators as functions.

        Usage : call_proc<functor, region>::[at<offseti, offsetj, offsetk>::]with[_offsets](eval, accessors...);

        Usage : call<functor, region>::[at_<offseti, offsetj, offsetk>::type::]with[_offsets](eval, accessors...);
    */
    template <typename Functor, typename Region, int Offi=0, int Offj=0, int Offk=0>
    struct call_proc {

        template <int I, int J, int K>
        struct at_ {
            typedef call_proc<Functor, Region, I, J, K> type;
        };

        template <int I, int J, int K>
        using at = call_proc<Functor, Region, I, J, K>;

        template <typename Evaluator, typename ...Args>
        GT_FUNCTION
        static
        void
        with(Evaluator const& eval, Args const & ...args) {

            typedef _impl::function_aggregator_procedure<
                Evaluator,
                Offi, Offj, Offk,
                typename _impl::package_args<Args...>::type
                > f_aggregator_t;

            auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

            Functor::Do
                (
                 f_aggregator_t
                 (
                  eval,
                  y
                  ),
                 Region()
                 );
        }

        template <typename Evaluator, typename ...Args>
        GT_FUNCTION
        static
        void
        with_offsets(Evaluator const& eval, Args const & ...args) {

            typedef _impl::function_aggregator_procedure_offsets<
                Evaluator,
                Offi, Offj, Offk,
                typename _impl::package_args<Args...>::type
                > f_aggregator_t;

            auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

            Functor::Do
                (
                 f_aggregator_t
                 (
                  eval,
                  y
                  ),
                 Region()
                 );
        }
};


} // namespace gridtools
