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

#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/vector.hpp>

#include "../../common/generic_metafunctions/mpl_sequence_to_fusion_vector.hpp"
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../accessor.hpp"
#include "../functor_decorator.hpp"
#include "../interval.hpp"           // to check if region is valid
#include "../iterate_domain_fwd.hpp" // to statically check arguments
#include "./call_interfaces_metafunctions.hpp"

namespace gridtools {
    // TODO: stencil functions work only for 3D stencils.

    namespace _impl {
        /** In the context of stencil_functions, this type represents
           the aggregator/domain/evaluator to be passed to a stencil
           function, called within a stencil operator or another
           stencil function. The accessors passed to
           the function can have offsets.

           function_aggregator_offsets has a single ReturnType which
           corresponds to the output field of the called
           stencil_operator. Such operator has a single output field,
           as checked by the call template.

           \tparam CallerAggregator The argument passed to the callerd, also known as the Evaluator
           \tparam Offi Offset along the i-direction were the function is evaluated (these are modified by specifying
        call<...>::at<...>::... )
           \tparam Offj Offset along the j-direction were the function is evaluated
           \tparam Offk Offset along the k-direction were the function is evaluated
           \tparam PassedArguments The list of accessors the caller need to pass to the function
           \tparam OutArg The index of the output argument of the function (this is required to be unique and it is
        check before this is instantiated.
        */
        template < typename CallerAggregator,
            int Offi,
            int Offj,
            int Offk,
            typename PassedArguments,
            typename ReturnType,
            int OutArg >
        struct function_aggregator_offsets {
            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< CallerAggregator >::value or is_function_aggregator< CallerAggregator >::value),
                "The first argument must be an iterate_domain or a function_aggregator");

            typedef typename boost::fusion::result_of::as_vector< PassedArguments >::type accessors_list_t;
            CallerAggregator &m_caller_aggregator;
            ReturnType *RESTRICT m_result;
            accessors_list_t const m_accessors_list;

            template < typename Accessor >
            struct accessor_return_type {
                typedef typename CallerAggregator::template accessor_return_type< Accessor >::type type;
            };

            GT_FUNCTION
            constexpr function_aggregator_offsets(
                CallerAggregator &caller_aggregator, ReturnType &result, accessors_list_t const &list)
                : m_caller_aggregator(caller_aggregator), m_result(&result), m_accessors_list(list) {}

            template < typename Accessor >
            using get_passed_argument_index =
                static_uint< (Accessor::index_t::value < OutArg) ? Accessor::index_t::value
                                                                 : (Accessor::index_t::value - 1) >;

            template < typename Accessor >
            using get_passed_argument_type =
                typename boost::mpl::at_c< PassedArguments, get_passed_argument_index< Accessor >::value >::type;

            template < typename Accessor >
            GT_FUNCTION constexpr auto get_passed_argument() const
                -> decltype(boost::fusion::at_c< get_passed_argument_index< Accessor >::value >(m_accessors_list)) {
                return boost::fusion::at_c< get_passed_argument_index< Accessor >::value >(m_accessors_list);
            }

            template < typename Accessor >
            using is_out_arg = boost::mpl::bool_< Accessor::index_t::value == OutArg >;

            /**
             * @brief Accessor is a normal 3D accessor
             */
            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::enable_if_c< not is_out_arg< Accessor >::value &&
                                                                   not is_global_accessor< Accessor >::value,
                typename accessor_return_type< get_passed_argument_type< Accessor > >::type >::type const
            operator()(Accessor const &accessor) const {
                GRIDTOOLS_STATIC_ASSERT((not is_global_accessor< get_passed_argument_type< Accessor > >::value),
                    "In call: you are passing a global_accessor to a normal accessor");
                return m_caller_aggregator(get_passed_argument_type< Accessor >(
                    accessor.template get< 2 >() + Offi + get_passed_argument< Accessor >().template get< 2 >(),
                    accessor.template get< 1 >() + Offj + get_passed_argument< Accessor >().template get< 1 >(),
                    accessor.template get< 0 >() + Offk + get_passed_argument< Accessor >().template get< 0 >()));
            }

            /**
             * @brief Accessor is a global_accessor
             */
            template < typename Accessor >
            GT_FUNCTION constexpr
                typename boost::enable_if_c< not is_out_arg< Accessor >::value && is_global_accessor< Accessor >::value,
                    typename accessor_return_type< get_passed_argument_type< Accessor > >::type >::type
                operator()(Accessor const &) const {
                GRIDTOOLS_STATIC_ASSERT((is_global_accessor< get_passed_argument_type< Accessor > >::value),
                    "In call: you are passing a normal accessor to a global_accessor");
                return m_caller_aggregator(get_passed_argument< Accessor >());
            }

            /**
             * @brief Accessor is the OutArg, just return the value
             */
            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::enable_if_c< is_out_arg< Accessor >::value, ReturnType >::type &
            operator()(Accessor const &) const {
                return *m_result;
            }

            template < typename... Arguments, template < typename... Args > class Expression >
            GT_FUNCTION constexpr auto operator()(Expression< Arguments... > const &arg) const
                -> decltype(expressions::evaluation::value(*this, arg)) {
                return expressions::evaluation::value((*this), arg);
            }
        };

        template < typename Functor, typename Region >
        struct do_caller {
            template < typename Aggregator >
            GT_FUNCTION static void Do(Aggregator &agg) {
                Functor::template Do< decltype(agg) & >(agg, Region());
            }
        };

        // overload for the default interval (Functor with one argument)
        template < typename Functor >
        struct do_caller< Functor, void > {
            template < typename Aggregator >
            GT_FUNCTION static void Do(Aggregator &agg) {
                Functor::template Do< decltype(agg) & >(agg);
            }
        };
    } // namespace _impl

    /** Main interface for calling stencil operators as functions.

        Usage C++11: call<functor, region>::[at<offseti, offsetj, offsetk>::]with(eval, accessors...);

        \tparam Functos The stencil operator to be called
        \tparam Region The region in which to call it (to take the proper overload). A region with no exact match is not
       called and will result in compilation error. The user is responsible for calling the proper Do overload)
        \tparam Offi Offset along the i-direction (usually modified using at<...>)
        \tparam Offj Offset along the j-direction
        \tparam Offk Offset along the k-direction
    */
    template < typename Functor, typename Region = void, int Offi = 0, int Offj = 0, int Offk = 0 >
    struct call {
        GRIDTOOLS_STATIC_ASSERT((is_interval< Region >::value or std::is_void< Region >::value),
            "Region should be a valid interval tag or void (default interval) to select the Do specialization in the "
            "called stencil function");

        /** This alias is used to move the computation at a certain offset
         */
        template < int I, int J, int K >
        using at = call< Functor, Region, I, J, K >;

      private:
        /**
           Obtain the result type of the function based on it's
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
        GT_FUNCTION static typename get_result_type< Evaluator, Functor >::type with(
            Evaluator &eval, Args const &... args) {

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

            auto agg_p = f_aggregator_t(eval, result, typename f_aggregator_t::accessors_list_t(args...));
            _impl::do_caller< Functor, Region >::Do(agg_p);

            return result;
        }
    };

    namespace _impl {
        /**
           In the context of stencil_functions, this type represents
           the aggregator/domain/evaluator to be passed to a stencil
           function, called within a stencil operator or another
           stencil function. The accessors passed to
           the function can have offsets.

           function_aggregator_procedure_offsets does not have a
           single return value, as in
           function_aggregator_offsets. Here there may be more than
           one returned values that happens through side-effects. The
           affected arguments are stored among the PassedArguments
           template argument.

           \tparam CallerAggregator The argument passed to the callerd, also known as the Evaluator
           \tparam Offi Offset along the i-direction were the function is evaluated (these are modified by
           specifying
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

            typedef typename boost::fusion::result_of::as_vector<
                typename mpl_sequence_to_fusion_vector< PassedArguments >::type >::type accessors_list_t;

            CallerAggregator &m_caller_aggregator;
            accessors_list_t const m_accessors_list;

            template < typename Accessor >
            struct accessor_return_type {
                typedef typename CallerAggregator::template accessor_return_type< Accessor >::type type;
            };

            template < typename Accessor >
            using passed_argument_is_accessor =
                is_accessor< typename boost::mpl::at_c< PassedArguments, Accessor::index_t::value >::type >;

            template < typename Accessor >
            using get_passed_argument_type =
                typename boost::mpl::at_c< PassedArguments, Accessor::index_t::value >::type;

            template < typename Accessor >
            GT_FUNCTION constexpr auto get_passed_argument() const
                -> decltype(boost::fusion::at_c< Accessor::index_t::value >(m_accessors_list)) {
                return boost::fusion::at_c< Accessor::index_t::value >(m_accessors_list);
            }

            GT_FUNCTION constexpr function_aggregator_procedure_offsets(
                CallerAggregator &caller_aggregator, accessors_list_t const &list)
                : m_caller_aggregator(caller_aggregator), m_accessors_list(list) {}

            /**
             * @brief Accessor is a normal 3D accessor (not a global_accessor) and the passed Argument is an accessor
             * (not a local variable)
             */
            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::enable_if_c< not is_global_accessor< Accessor >::value &&
                                                                   passed_argument_is_accessor< Accessor >::value,
                typename CallerAggregator::template accessor_return_type<
                    get_passed_argument_type< Accessor > >::type >::type
            operator()(Accessor const &accessor) const {
                GRIDTOOLS_STATIC_ASSERT((not is_global_accessor< get_passed_argument_type< Accessor > >::value),
                    "In call_proc: you are passing a global_accessor to a normal accessor");
                return m_caller_aggregator(get_passed_argument_type< Accessor >(
                    accessor.template get< 2 >() + Offi + get_passed_argument< Accessor >().template get< 2 >(),
                    accessor.template get< 1 >() + Offj + get_passed_argument< Accessor >().template get< 1 >(),
                    accessor.template get< 0 >() + Offk + get_passed_argument< Accessor >().template get< 0 >()));
            }

            /**
             * @brief Accessor is a global_accessor and the passed Argument is an accessor (not a local variable)
             */
            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::enable_if_c< is_global_accessor< Accessor >::value &&
                                                                   passed_argument_is_accessor< Accessor >::value,
                typename CallerAggregator::template accessor_return_type<
                                                                   get_passed_argument_type< Accessor > >::type >::type
            operator()(Accessor const &accessor) const {
                GRIDTOOLS_STATIC_ASSERT((is_global_accessor< get_passed_argument_type< Accessor > >::value),
                    "In call_proc: you are passing a normal accessor to a global_accessor");
                return m_caller_aggregator(get_passed_argument< Accessor >());
            }

            /**
             * @brief Passed argument is a local variable (not an accessor)
             */
            template < typename Accessor >
            GT_FUNCTION constexpr typename boost::enable_if_c< not passed_argument_is_accessor< Accessor >::value,
                typename boost::remove_reference< typename boost::fusion::result_of::at_c< accessors_list_t,
                    Accessor::index_t::value >::type >::type::type >::type &
            operator()(Accessor const &) const {
                return get_passed_argument< Accessor >().value();
            }

            template < typename... Arguments, template < typename... Args > class Expression >
            GT_FUNCTION constexpr auto operator()(Expression< Arguments... > const &arg) const
                -> decltype(expressions::evaluation::value(*this, arg)) {
                return expressions::evaluation::value((*this), arg);
            }
        };
    } // namespace _impl

    /** Main interface for calling stencil operators as functions with
        side-effects. The interface accepts a list of arguments to be
        passed to the called function and these arguments can be
        accessors or simple values.

        Usage : call_proc<functor, region>::[at<offseti, offsetj, offsetk>::]with(eval, accessors_or_values...);

        Accessors_or_values referes to a list of arguments that may be
        accessors of the caller functions or local variables of the
        type accessed (or converted to) by the accessor in the called
        function, where the results should be obtained from. The
        values can also be used by the function as inputs.

        \tparam Functos The stencil operator to be called
        \tparam Region The region in which to call it (to take the proper overload). A region with no exact match is
       not
       called and will result in compilation error. The user is responsible for calling the proper Do overload)
        \tparam Offi Offset along the i-direction (usually modified using at<...>)
        \tparam Offj Offset along the j-direction
        \tparam Offk Offset along the k-direction
    */
    template < typename Functor, typename Region = void, int Offi = 0, int Offj = 0, int Offk = 0 >
    struct call_proc {

        GRIDTOOLS_STATIC_ASSERT((is_interval< Region >::value or std::is_void< Region >::value),
            "Region should be a valid interval tag or void (default interval) to select the Do specialization in the "
            "called stencil function");

        /** This alias is used to move the computation at a certain offset
         */
        template < int I, int J, int K >
        using at = call_proc< Functor, Region, I, J, K >;

        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are used to
            access values, w.r.t the offsets specified in a optional
            at<..> statement.
         */
        template < typename Evaluator, typename... Args >
        GT_FUNCTION static void with(Evaluator &eval, Args const &... args) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain< Evaluator >::value or _impl::is_function_aggregator< Evaluator >::value),
                "The first argument must be the Evaluator/Aggregator of the stencil operator.");

            typedef _impl::function_aggregator_procedure_offsets< Evaluator,
                Offi,
                Offj,
                Offk,
                typename _impl::package_args< Args... >::type > f_aggregator_t;

            auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

            auto agg_p = f_aggregator_t(eval, y);
            _impl::do_caller< Functor, Region >::Do(agg_p);
        }
    };
} // namespace gridtools
