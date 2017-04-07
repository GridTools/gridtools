/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include "./accessor.hpp"
#include "./call_interfaces_metafunctions.hpp"
#include "../../common/generic_metafunctions/mpl_sequence_to_fusion_vector.hpp"
#include "../iterate_domain_fwd.hpp" // to statically check arguments
#include "../interval.hpp"           // to check if region is valid
#include "../functor_decorator.hpp"

namespace gridtools {
    // TODO: stencil functions works only for 3D stencils.

    namespace _impl {
        /**
           In the context of stencil_functions, this type represents
           the aggregator/domain/evaluator to be passed to a stencil
           function, called within a stencil operator or another
           stencil function.  The offsets of the accessors passed here
           must be 0, but are otherwise ignored.

           function_aggregator has a single ReturnType which
           corresponds to the output field of the called
           stencil_operator. Such operator hasa single
           output field, as checked by the call template.

           \tparam CallerAggregator The argument passed to the callerd, also known as the Evaluator
           \tparam Offi Offset along the i-direction were the function is evaluated (these are modified by specifying
           call<...>::at<...>::... )
           \tparam Offj Offset along the j-direction were the function is evaluated
           \tparam Offk Offset along the k-direction were the function is evaluated
           \tparam PassedAccessors The list of accessors the caller need to pass to the function
           \tparam OutArg The index of the output argument of the function (this is required to be unique and it is
           check before this is instantiated.
        */
        template < typename CallerAggregator,
            int Offi,
            int Offj,
            int Offk,
            typename PassedAccessors,
            typename ReturnType,
            int OutArg >
        struct function_aggregator {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< CallerAggregator >::value or is_function_aggregator< CallerAggregator >::value),
                "The first argument must be an iterate_domain or a function_aggregator");

            CallerAggregator const &m_caller_aggregator;
            ReturnType *__restrict__ m_result;

            template < typename Accessor >
            struct accessor_return_type {
                typedef typename CallerAggregator::template accessor_return_type< Accessor >::type type;
            };

            GT_FUNCTION
            function_aggregator(CallerAggregator const &caller_aggregator, ReturnType &result)
                : m_caller_aggregator(caller_aggregator), m_result(&result) {}

            template < typename Accessor >
            GT_FUNCTION constexpr
                typename boost::enable_if_c< (Accessor::index_type::value < OutArg), ReturnType >::type const
                operator()(Accessor const &accessor) const {
                return m_caller_aggregator(
                    typename boost::mpl::at_c< PassedAccessors, Accessor::index_type::value >::type(
                        accessor.template get< 2 >() + Offi,
                        accessor.template get< 1 >() + Offj,
                        accessor.template get< 0 >() + Offk));
            }

            template < typename Accessor >
            GT_FUNCTION constexpr
                typename boost::enable_if_c< (Accessor::index_type::value > OutArg), ReturnType >::type const
                operator()(Accessor const &accessor) const {
                return m_caller_aggregator(
                    typename boost::mpl::at_c< PassedAccessors, Accessor::index_type::value - 1 >::type(
                        accessor.template get< 2 >() + Offi,
                        accessor.template get< 1 >() + Offj,
                        accessor.template get< 0 >() + Offk));
            }

            template < typename Accessor >
            GT_FUNCTION constexpr
                typename boost::enable_if_c< (Accessor::index_type::value == OutArg), ReturnType >::type &
                operator()(Accessor const &) const {
                // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
                return *m_result;
            }

            /** @brief method called in the Do methods of the functors. */
            template < typename... Arguments, template < typename... Args > class Expression >
            GT_FUNCTION constexpr auto operator()(Expression< Arguments... > const &arg) const
                -> decltype(expressions::evaluation::value(*this, arg)) {
                // arg.to_string();
                return expressions::evaluation::value((*this), arg);
            }

            /** @brief method called in the Do methods of the functors.
                partial specializations for double (or float)*/
            template < typename Accessor,
                template < typename Arg1, typename Arg2 > class Expression,
                typename FloatType,
                typename boost::enable_if< typename boost::is_floating_point< FloatType >::type, int >::type = 0 >
            GT_FUNCTION constexpr auto operator()(Expression< Accessor, FloatType > const &arg) const
                -> decltype(expressions::evaluation::value(*this, arg)) {
                // TODO RENAME ACCESSOR,is not an accessor but an expression, and add an assertion for type
                return expressions::evaluation::value((*this), arg);
            }
        };

        /**
        /** In the context of stencil_functions, this type represents
           the aggregator/domain/evaluator to be passed to a stencil
           function, called within a stencil operator or another
           stencil function. In this version the accessors passed to
           the function can have offsets different that 0.

           function_aggregator_offsets has a single ReturnType which
           corresponds to the output field of the called
           stencil_operator. Such operator hasa single output field,
           as checked by the call template.

           \tparam CallerAggregator The argument passed to the callerd, also known as the Evaluator
           \tparam Offi Offset along the i-direction were the function is evaluated (these are modified by specifying
        call<...>::at<...>::... )
           \tparam Offj Offset along the j-direction were the function is evaluated
           \tparam Offk Offset along the k-direction were the function is evaluated
           \tparam PassedAccessors The list of accessors the caller need to pass to the function
           \tparam OutArg The index of the output argument of the function (this is required to be unique and it is
        check before this is instantiated.
        */
        template < typename CallerAggregator,
            int Offi,
            int Offj,
            int Offk,
            typename PassedAccessors,
            typename ReturnType,
            int OutArg >
        struct function_aggregator_offsets {
            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< CallerAggregator >::value or is_function_aggregator< CallerAggregator >::value),
                "The first argument must be an iterate_domain or a function_aggregator");

            typedef typename boost::fusion::result_of::as_vector< PassedAccessors >::type accessors_list_t;
            CallerAggregator const &m_caller_aggregator;
            ReturnType *__restrict__ m_result;
            accessors_list_t const &m_accessors_list;

            template < typename Accessor >
            struct accessor_return_type {
                typedef typename CallerAggregator::template accessor_return_type< Accessor >::type type;
            };

            GT_FUNCTION
            constexpr function_aggregator_offsets(
                CallerAggregator const &caller_aggregator, ReturnType &result, accessors_list_t const &list)
                : m_caller_aggregator(caller_aggregator), m_result(&result), m_accessors_list(list) {}

            template < typename Accessor >
            GT_FUNCTION constexpr
                typename boost::enable_if_c< (Accessor::index_type::value < OutArg), ReturnType >::type const
                operator()(Accessor const &accessor) const {
                return m_caller_aggregator(
                    typename boost::mpl::at_c< PassedAccessors, Accessor::index_type::value >::type(
                        accessor.template get< 2 >() + Offi +
                            boost::fusion::at_c< Accessor::index_type::value >(m_accessors_list).template get< 2 >(),
                        accessor.template get< 1 >() + Offj +
                            boost::fusion::at_c< Accessor::index_type::value >(m_accessors_list).template get< 1 >(),
                        accessor.template get< 0 >() + Offk +
                            boost::fusion::at_c< Accessor::index_type::value >(m_accessors_list).template get< 0 >()));
            }

            template < typename Accessor >
            GT_FUNCTION constexpr
                typename boost::enable_if_c< (Accessor::index_type::value > OutArg), ReturnType >::type const
                operator()(Accessor const &accessor) const {
                return m_caller_aggregator(
                    typename boost::mpl::at_c< PassedAccessors, Accessor::index_type::value - 1 >::type(
                        accessor.template get< 2 >() + Offi +
                            boost::fusion::at_c< Accessor::index_type::value - 1 >(m_accessors_list)
                                .template get< 2 >(),
                        accessor.template get< 1 >() + Offj +
                            boost::fusion::at_c< Accessor::index_type::value - 1 >(m_accessors_list)
                                .template get< 1 >(),
                        accessor.template get< 0 >() + Offk +
                            boost::fusion::at_c< Accessor::index_type::value - 1 >(m_accessors_list)
                                .template get< 0 >()));
            }

            template < typename Accessor >
            GT_FUNCTION constexpr
                typename boost::enable_if_c< (Accessor::index_type::value == OutArg), ReturnType >::type &
                operator()(Accessor const &) const {
                // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
                return *m_result;
            }
        };
    } // namespace _impl

    /** wrap the regular interval in this identity metafunction, for doing lazy type instantiation in the
     * boost::mpl::if_*/
    template < typename T >
    struct wrap_default_interval {
        typedef T default_interval;
    };

    /** Main interface for calling stencil operators as functions.

        Usage C++11: call<functor, region>::[at<offseti, offsetj, offsetk>::]with(eval, accessors...);

        Usage : call<functor, region>::[at<offseti, offsetj, offsetk>::type::]with(eval, accessors...);

        \tparam Functos The stencil operator to be called
        \tparam Region The region in which to call it (to take the proper overload). A region with no exact match is not
       called and will result in compilation error. The user is responsible for calling the proper Do overload)
        \tparam Offi Offset along the i-direction (usually modified using at<...>)
        \tparam Offj Offset along the j-direction
        \tparam Offk Offset along the k-direction
    */
    template < typename Functor, typename Region = void, int Offi = 0, int Offj = 0, int Offk = 0 >
    struct call {

        GRIDTOOLS_STATIC_ASSERT((sfinae::has_two_args< Functor >::value),
            "you cannot pass a specific vertical "
            "interval to call a Do method with only one "
            "argument (i.e. using the default interval)");
        GRIDTOOLS_STATIC_ASSERT((is_interval< Region >::value),
            "Region should be a valid interval tag to select the Do specialization in the called stencil function,");

        /** This alias is used to move the computation at a certain offset
         */
        template < int I, int J, int K >
        using at = call< Functor, Region, I, J, K >;

      private:
        /**
           Obtain the result tyoe of the function based on it's
           signature
         */
        template < typename Eval, typename Funct >
        struct get_result_type {
            typedef accessor< _impl::_get_index_of_first_non_const< Funct >::value > accessor_t;

            typedef typename Eval::template accessor_return_type< accessor_t >::type r_type;

            typedef typename std::decay< r_type >::type type;
        };

      public:
        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are used to
            access values, w.r.t the offsets specified in a optional
            at<..> statement.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static typename get_result_type< Evaluator, Functor >::type with_offsets(
            Evaluator const &eval, Args const &... args) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            GRIDTOOLS_STATIC_ASSERT(_impl::can_be_a_function< Functor >::value,
                "Trying to invoke stencil operator with more than one output as a function\n");

            typedef typename get_result_type< Evaluator, Functor >::type result_type;
            typedef _impl::function_aggregator_offsets< Evaluator,
                Offi,
                Offj,
                Offk,
                typename gridtools::variadic_to_vector< Args... >::type,
                result_type,
                _impl::_get_index_of_first_non_const< Functor >::value > f_aggregator_t;

            result_type result;

            Functor::Do(f_aggregator_t(eval, result, typename f_aggregator_t::accessors_list_t(args...)), Region());

            return result;
        }

        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are ignored.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static typename get_result_type< Evaluator, Functor >::type with(
            Evaluator const &eval, Args const &...) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            GRIDTOOLS_STATIC_ASSERT(_impl::can_be_a_function< Functor >::value,
                "Trying to invoke stencil operator with more than one output as a function\n");

            typedef typename get_result_type< Evaluator, Functor >::type result_type;

            result_type result;
            typedef _impl::function_aggregator< Evaluator,
                Offi,
                Offj,
                Offk,
                typename gridtools::variadic_to_vector< Args... >::type,
                result_type,
                _impl::_get_index_of_first_non_const< Functor >::value > f_aggregator_t;

            Functor::Do(f_aggregator_t(eval, result), Region());

            return result;
        }
    };

    /**
       \brief specialization for when we call a stencil function without specifiyng any interval (using as default
       interval the whole axis).

       The main restriction is that the compile-time offsets template parameters must be 0
       NOTE: There is some code replication with the primary template class
     */
    template < typename Functor, int Offi, int Offj >
    struct call< Functor, void, Offi, Offj, 0 > {

        /** This alias is used to move the computation at a certain offset
         */
        template < int I, int J, int K >
        struct at : call< Functor, void, I, J, 0 > {
            GRIDTOOLS_STATIC_ASSERT((K == 0),
                "Cannot specify a nonzero vertical offset (with the call::at API) in a "
                "stencil function call with default interval: would go out of bounds");
        };

      private:
        /**
           Obtain the result tyoe of the function based on it's
           signature
         */
        template < typename Eval, typename Funct >
        struct get_result_type {
            typedef accessor< _impl::_get_index_of_first_non_const< Funct >::value > accessor_t;

            typedef typename Eval::template accessor_return_type< accessor_t >::type r_type;

            typedef typename std::decay< r_type >::type type;
        };

      public:
        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are used to
            access values, w.r.t the offsets specified in a optional
            at<..> statement.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static typename get_result_type< Evaluator, Functor >::type with_offsets(
            Evaluator const &eval, Args const &... args) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            GRIDTOOLS_STATIC_ASSERT(_impl::can_be_a_function< Functor >::value,
                "Trying to invoke stencil operator with more than one output as a function\n");

            typedef typename get_result_type< Evaluator, Functor >::type result_type;
            typedef _impl::function_aggregator_offsets< Evaluator,
                Offi,
                Offj,
                0,
                typename gridtools::variadic_to_vector< Args... >::type,
                result_type,
                _impl::_get_index_of_first_non_const< Functor >::value > f_aggregator_t;

            result_type result;

            Functor::Do(f_aggregator_t(eval, result, typename f_aggregator_t::accessors_list_t(args...)));

            return result;
        }

        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are ignored.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static typename get_result_type< Evaluator, Functor >::type with(
            Evaluator const &eval, Args const &...) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            GRIDTOOLS_STATIC_ASSERT(_impl::can_be_a_function< Functor >::value,
                "Trying to invoke stencil operator with more than one output as a function\n");

            typedef typename get_result_type< Evaluator, Functor >::type result_type;

            result_type result;
            typedef _impl::function_aggregator< Evaluator,
                Offi,
                Offj,
                0,
                typename gridtools::variadic_to_vector< Args... >::type,
                result_type,
                _impl::_get_index_of_first_non_const< Functor >::value > f_aggregator_t;

            Functor::Do(f_aggregator_t(eval, result));

            return result;
        }
    };

    namespace _impl {
        /**
           In the context of stencil_functions, this type represents
           the aggregator/domain/evaluator to be passed to a stencil
           function, called within a stencil operator or another
           stencil function. In this version the accessors passed to
           the function can have offsets different that 0.

           function_aggregator_procedure_offsets does not have a
           single return value, as in
           function_aggregator_offsets. Here there may be more than
           one returned values that happens through side-effects. The
           affected arguments are stored among the PassedArguments
           template argument.

           \tparam CallerAggregator The argument passed to the callerd, also known as the Evaluator
           \tparam Offi Offset along the i-direction were the function is evaluated (these are modified by specifying
           call<...>::at<...>::... )
           \tparam Offj Offset along the j-direction were the function is evaluated
           \tparam Offk Offset along the k-direction were the function is evaluated
           \tparam PassedArguments The list of accessors and other orguments the caller need to pass to the function
        */
        template < typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedArguments >
        struct function_aggregator_procedure_offsets {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< CallerAggregator >::value or is_function_aggregator< CallerAggregator >::value),
                "The first argument must be an iterate_domain or a function_aggregator");

            // Collect the indices of the arguments that are not accessors among
            // the PassedArguments
            typedef
                typename boost::mpl::fold< boost::mpl::range_c< int, 0, boost::mpl::size< PassedArguments >::value >,
                    boost::mpl::vector0<>,
                    typename _impl::insert_index_if_not_accessor< PassedArguments >::template apply< boost::mpl::_2,
                                               boost::mpl::_1 > >::type non_accessor_indices;

            //        typedef typename wrap_reference<PassedArguments>::type wrapped_accessors
            // typedef typename boost::fusion::result_of::as_vector<wrapped_accessors>::type accessors_list_t;
            typedef typename boost::fusion::result_of::as_vector<
                typename mpl_sequence_to_fusion_vector< PassedArguments >::type >::type accessors_list_t;

            CallerAggregator const &m_caller_aggregator;
            accessors_list_t const &m_accessors_list;

            template < typename Accessor >
            struct accessor_return_type {
                typedef typename CallerAggregator::template accessor_return_type< Accessor >::type type;
            };

            GT_FUNCTION
            constexpr function_aggregator_procedure_offsets(
                CallerAggregator const &caller_aggregator, accessors_list_t const &list)
                : m_caller_aggregator(caller_aggregator), m_accessors_list(list) {}

            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::enable_if_c<
                not _impl::contains_value< non_accessor_indices, typename Accessor::index_type >::value,
                typename CallerAggregator::template accessor_return_type<
                    typename boost::mpl::at_c< PassedArguments, Accessor::index_type::value >::type >::type >::type
            operator()(Accessor const &accessor) const {
                return m_caller_aggregator(
                    typename boost::mpl::at_c< PassedArguments, Accessor::index_type::value >::type(
                        accessor.template get< 2 >() + Offi +
                            boost::fusion::at_c< Accessor::index_type::value >(m_accessors_list).template get< 2 >(),
                        accessor.template get< 1 >() + Offj +
                            boost::fusion::at_c< Accessor::index_type::value >(m_accessors_list).template get< 1 >(),
                        accessor.template get< 0 >() + Offk +
                            boost::fusion::at_c< Accessor::index_type::value >(m_accessors_list).template get< 0 >()));
            }

            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::enable_if_c<
                _impl::contains_value< non_accessor_indices, typename Accessor::index_type >::value,
                typename boost::remove_reference< typename boost::fusion::result_of::at_c< accessors_list_t,
                    Accessor::index_type::value >::type >::type::type >::type &
            operator()(Accessor const &) const {
                // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
                return (boost::fusion::at_c< Accessor::index_type::value >(m_accessors_list).value());
            }
        };

        /**
           In the context of stencil_functions, this type represents
           the aggregator/domain/evaluator to be passed to a stencil
           function, called within a stencil operator or another
           stencil function. The offsets of the accessors passed here
           must be 0, but are otherwise ignored.

           function_aggregator_procedure does not have a single return
           value, as in function_aggregator. Here there may be more
           than one returned values that happens through
           side-effects. The affected arguments are stored among the
           PassedArguments template argument.

           \tparam CallerAggregator The argument passed to the callerd, also known as the Evaluator
           \tparam Offi Offset along the i-direction were the function is evaluated (these are modified by specifying
           call<...>::at<...>::... )
           \tparam Offj Offset along the j-direction were the function is evaluated
           \tparam Offk Offset along the k-direction were the function is evaluated
           \tparam PassedArguments The list of accessors and other orguments the caller need to pass to the function
        */
        template < typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedArguments >
        struct function_aggregator_procedure {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< CallerAggregator >::value or is_function_aggregator< CallerAggregator >::value),
                "The first argument must be an iterate_domain or a function_aggregator");

            // Collect the indices of the arguments that are not accessors among
            // the PassedArguments
            typedef
                typename boost::mpl::fold< boost::mpl::range_c< int, 0, boost::mpl::size< PassedArguments >::value >,
                    boost::mpl::vector0<>,
                    typename _impl::insert_index_if_not_accessor< PassedArguments >::template apply< boost::mpl::_2,
                                               boost::mpl::_1 > >::type non_accessor_indices;

            typedef typename boost::fusion::result_of::as_vector<
                typename mpl_sequence_to_fusion_vector< PassedArguments >::type >::type accessors_list_t;

            CallerAggregator const &m_caller_aggregator;
            accessors_list_t const &m_accessors_list;

            template < typename Accessor >
            struct accessor_return_type {
                typedef typename CallerAggregator::template accessor_return_type< Accessor >::type type;
            };

            GT_FUNCTION
            function_aggregator_procedure(CallerAggregator const &caller_aggregator, accessors_list_t const &list)
                : m_caller_aggregator(caller_aggregator), m_accessors_list(list) {}

            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::lazy_enable_if_c<
                not _impl::contains_value< non_accessor_indices, typename Accessor::index_type >::value,
                typename CallerAggregator::template accessor_return_type<
                    typename boost::mpl::at_c< PassedArguments, Accessor::index_type::value >::type > //::type
                >::type
            operator()(Accessor const &accessor) const {
                return m_caller_aggregator(
                    typename boost::mpl::at_c< PassedArguments, Accessor::index_type::value >::type(
                        accessor.template get< 2 >() + Offi,
                        accessor.template get< 1 >() + Offj,
                        accessor.template get< 0 >() + Offk));
            }

            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::lazy_enable_if_c<
                _impl::contains_value< non_accessor_indices, typename Accessor::index_type >::value,
                typename boost::remove_reference< typename boost::fusion::result_of::at_c< accessors_list_t,
                    Accessor::index_type::value >::type >::type //::type
                >::type &
            operator()(Accessor const &) const {
                // std::cout << "Giving the ref (OutArg=" << OutArg << ") " << m_result << std::endl;
                return (boost::fusion::at_c< Accessor::index_type::value >(m_accessors_list).value());
            }
        };

    } // namespace _impl

    /** Main interface for calling stencil operators as functions with
        side-effects. The interface accepts a list of arguments to be
        passed to the called function and these arguments can be
        accessors or simple values.

        Usage : call_proc<functor, region>::[at<offseti, offsetj, offsetk>::]with[_offsets](eval,
       accessors_or_values...);

        Usage : call<functor, region>::[at_<offseti, offsetj, offsetk>::type::]with[_offsets](eval,
       accessors_of_values...);

        Accessors_or_values referes to a list of arguments that may be
        accessors of the caller functions or local variables of the
        type accessed (or converted to) by the accessor in the called
        function, where the results should be obtained from. The
        values can also be used by the function as inputs.

        \tparam Functos The stencil operator to be called
        \tparam Region The region in which to call it (to take the proper overload). A region with no exact match is not
       called and will result in compilation error. The user is responsible for calling the proper Do overload)
        \tparam Offi Offset along the i-direction (usually modified using at<...>)
        \tparam Offj Offset along the j-direction
        \tparam Offk Offset along the k-direction
    */
    template < typename Functor, typename Region = void, int Offi = 0, int Offj = 0, int Offk = 0 >
    struct call_proc {

        GRIDTOOLS_STATIC_ASSERT((is_interval< Region >::value),
            "Region should be a valid interval tag to select the Do specialization in the called stencil function,");

        /** This alias is used to move the computation at a certain offset
         */
        template < int I, int J, int K >
        using at = call_proc< Functor, Region, I, J, K >;

        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are ignored.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static void with(Evaluator const &eval, Args const &... args) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            typedef _impl::function_aggregator_procedure< Evaluator,
                Offi,
                Offj,
                Offk,
                typename _impl::package_args< Args... >::type > f_aggregator_t;

            auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

            Functor::Do(f_aggregator_t(eval, y), Region());
        }

        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are used to
            access values, w.r.t the offsets specified in a optional
            at<..> statement.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static void with_offsets(Evaluator const &eval, Args const &... args) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            typedef _impl::function_aggregator_procedure_offsets< Evaluator,
                Offi,
                Offj,
                Offk,
                typename _impl::package_args< Args... >::type > f_aggregator_t;

            auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

            Functor::Do(f_aggregator_t(eval, y), Region());
        }
    };

    /**
       \brief specialization for when we call a stencil function without specifiyng any interval (using as default
       interval the whole axis).

       The main restriction is that the compile-time offsets template parameters must be 0
       NOTE: There is some code replication with the primary template class
     */
    template < typename Functor, int Offi, int Offj >
    struct call_proc< Functor, void, Offi, Offj, 0 > {

        /** This alias is used to move the computation at a certain offset
         */
        template < int I, int J, int K >
        struct at : call_proc< Functor, void, I, J, 0 > {
            GRIDTOOLS_STATIC_ASSERT((K == 0),
                "Cannot specify a nonzero vertical offset (with the call::at API) in a "
                "stencil function call with default interval: would go out of bounds");
        };

        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are ignored.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static void with(Evaluator const &eval, Args const &... args) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            typedef _impl::
                function_aggregator_procedure< Evaluator, Offi, Offj, 0, typename _impl::package_args< Args... >::type >
                    f_aggregator_t;

            auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

            Functor::Do(f_aggregator_t(eval, y));
        }

        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are used to
            access values, w.r.t the offsets specified in a optional
            at<..> statement.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static void with_offsets(Evaluator const &eval, Args const &... args) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            typedef _impl::function_aggregator_procedure_offsets< Evaluator,
                Offi,
                Offj,
                0,
                typename _impl::package_args< Args... >::type > f_aggregator_t;

            auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

            Functor::Do(f_aggregator_t(eval, y));
        }
    };

} // namespace gridtools
