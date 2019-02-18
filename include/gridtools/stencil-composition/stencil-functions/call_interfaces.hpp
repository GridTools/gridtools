/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/vector.hpp>
#include <type_traits>

#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../accessor.hpp"
#include "../expressions/expr_base.hpp"
#include "../interval.hpp" // to check if region is valid
#include "../offset_computation.hpp"
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
           corresponds to the type of the first argument in the call.
           Such operator has a single output field,
           as checked by the call template.

           \tparam CallerAggregator The argument passed to the caller, also known as the Evaluator
           \tparam Offi Offset along the i-direction were the function is evaluated (these are modified by specifying
        call<...>::at<...>::... )
           \tparam Offj Offset along the j-direction were the function is evaluated
           \tparam Offk Offset along the k-direction were the function is evaluated
           \tparam PassedArguments The list of accessors the caller need to pass to the function
           \tparam OutArg The index of the output argument of the function (this is required to be unique and it is
        check before this is instantiated.
        */
        template <typename CallerAggregator,
            int Offi,
            int Offj,
            int Offk,
            typename PassedArguments,
            typename ReturnType,
            int OutArg>
        struct function_aggregator_offsets {
            typedef typename boost::fusion::result_of::as_vector<PassedArguments>::type accessors_list_t;
            CallerAggregator &m_caller_aggregator;
            ReturnType *GT_RESTRICT m_result;
            accessors_list_t const m_accessors_list;

            template <typename Accessor>
            using get_passed_argument_index =
                static_uint<(Accessor::index_t::value < OutArg) ? Accessor::index_t::value
                                                                : (Accessor::index_t::value - 1)>;

            template <typename Accessor>
            using get_passed_argument_t =
                typename boost::mpl::at_c<PassedArguments, get_passed_argument_index<Accessor>::value>::type;

            template <typename Accessor>
            GT_FUNCTION constexpr get_passed_argument_t<Accessor> get_passed_argument() const {
                return boost::fusion::at_c<get_passed_argument_index<Accessor>::value>(m_accessors_list);
            }

            template <typename Accessor>
            using passed_argument_is_accessor_t = typename is_accessor<get_passed_argument_t<Accessor>>::type;

            template <typename Accessor>
            using is_out_arg = boost::mpl::bool_<Accessor::index_t::value == OutArg>;

          public:
            GT_FUNCTION
            constexpr function_aggregator_offsets(
                CallerAggregator &caller_aggregator, ReturnType &result, accessors_list_t const &list)
                : m_caller_aggregator(caller_aggregator), m_result(&result), m_accessors_list(list) {}

            /**
             * @brief Accessor (of the callee) is a regular 3D in_accessor
             */
            template <typename Accessor,
                enable_if_t<passed_argument_is_accessor_t<Accessor>::value && not is_out_arg<Accessor>::value &&
                                not is_global_accessor<Accessor>::value,
                    int> = 0>
            GT_FUNCTION constexpr auto operator()(Accessor const &accessor) const
                -> decltype(m_caller_aggregator(get_passed_argument_t<Accessor>())) {
                return m_caller_aggregator(get_passed_argument_t<Accessor>(
                    accessor_offset<0>(accessor) + Offi + accessor_offset<0>(get_passed_argument<Accessor>()),
                    accessor_offset<1>(accessor) + Offj + accessor_offset<1>(get_passed_argument<Accessor>()),
                    accessor_offset<2>(accessor) + Offk + accessor_offset<2>(get_passed_argument<Accessor>())));
            }

            /*
             * @brief If the passed type is not an accessor we assume it is a local variable which we just return.
             */
            template <typename Accessor>
            GT_FUNCTION constexpr enable_if_t<not passed_argument_is_accessor_t<Accessor>::value &&
                                                  not is_out_arg<Accessor>::value,
                get_passed_argument_t<Accessor>>
            operator()(Accessor const &accessor) const {
                return get_passed_argument<Accessor>();
            }

            /**
             * @brief Accessor is a global_accessor
             */
            template <typename Accessor,
                enable_if_t<passed_argument_is_accessor_t<Accessor>::value && not is_out_arg<Accessor>::value &&
                                is_global_accessor<Accessor>::value,
                    int> = 0>
            GT_FUNCTION constexpr auto operator()(Accessor const &) const
                -> decltype(m_caller_aggregator(get_passed_argument_t<Accessor>())) {
                return m_caller_aggregator(get_passed_argument<Accessor>());
            }

            /**
             * @brief Accessor is the (only!) OutArg, i.e. the return value
             */
            template <typename Accessor>
            GT_FUNCTION constexpr typename std::enable_if<is_out_arg<Accessor>::value, ReturnType>::type &operator()(
                Accessor const &) const {
                return *m_result;
            }

            template <class Op, class... Args>
            GT_FUNCTION constexpr auto operator()(expr<Op, Args...> const &arg) const
                GT_AUTO_RETURN(expressions::evaluation::value(*this, arg));
        };

        template <class Functor, class Region, class Eval>
        GT_FUNCTION enable_if_t<!std::is_void<Region>::value> call_functor(Eval &eval) {
            Functor::template apply<Eval &>(eval, Region{});
        }

        // overload for the default interval (Functor with one argument)
        template <class Functor, class Region, class Eval>
        GT_FUNCTION enable_if_t<std::is_void<Region>::value> call_functor(Eval &eval) {
            Functor::template apply<Eval &>(eval);
        }
    } // namespace _impl

    /** Main interface for calling stencil operators as functions.

        Usage: call<functor, region>::[return_type<>::][at<offseti, offsetj, offsetk>::]with(eval, accessors...);

        \tparam Functos The stencil operator to be called
        \tparam Region The region in which to call it (to take the proper overload). A region with no exact match is not
       called and will result in compilation error. The user is responsible for calling the proper apply overload)
        \tparam ReturnType Can be set or will be deduced from the first input argument
        \tparam Offi Offset along the i-direction (usually modified using at<...>)
        \tparam Offj Offset along the j-direction
        \tparam Offk Offset along the k-direction
    */
    template <typename Functor,
        typename Region = void,
        typename ReturnType = void,
        int Offi = 0,
        int Offj = 0,
        int Offk = 0>
    struct call {
        GT_STATIC_ASSERT((is_interval<Region>::value or std::is_void<Region>::value),
            "Region should be a valid interval tag or void (default interval) to select the apply specialization in "
            "the called stencil function");

        /** This alias is used to move the computation at a certain offset
         */
        template <int I, int J, int K>
        using at = call<Functor, Region, ReturnType, I, J, K>;

        /**
         * @brief alias to set the return type, e.g.
         */
        template <typename ForcedReturnType>
        using return_type = call<Functor, Region, ForcedReturnType, Offi, Offj, Offk>;

      private:
        template <typename Eval, typename Arg, bool = is_accessor<Arg>::value>
        struct decude_result_type : std::decay<decltype(std::declval<Eval &>()(std::declval<Arg const &>()))> {};

        template <typename Eval, typename Arg>
        struct decude_result_type<Eval, Arg, false> : meta::lazy::id<Arg> {};

        /**
         * @brief Use forced return type (if not void) or deduce the return type.
         */
        template <typename Eval, typename Arg, typename...>
        struct get_result_type : std::conditional<std::is_void<ReturnType>::value,
                                     typename decude_result_type<Eval, Arg>::type,
                                     ReturnType> {};

      public:
        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are used to
            access values, w.r.t the offsets specified in a optional
            at<..> statement.
         */
        template <typename Evaluator, typename... Args>
        GT_FUNCTION static typename get_result_type<Evaluator, Args...>::type with(
            Evaluator &eval, Args const &... args) {

            GT_STATIC_ASSERT(_impl::can_be_a_function<Functor>::value,
                "Trying to invoke stencil operator with more than one output as a function\n");

            typedef typename get_result_type<Evaluator, Args...>::type result_type;
            typedef _impl::function_aggregator_offsets<Evaluator,
                Offi,
                Offj,
                Offk,
                typename gridtools::variadic_to_vector<Args...>::type,
                result_type,
                _impl::_get_index_of_first_non_const<Functor>::value>
                f_aggregator_t;

            result_type result;

            auto agg_p = f_aggregator_t(eval, result, typename f_aggregator_t::accessors_list_t(args...));
            _impl::call_functor<Functor, Region>(agg_p);

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
        template <typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedArguments>
        struct function_aggregator_procedure_offsets {
            typedef typename boost::fusion::result_of::as_vector<PassedArguments>::type accessors_list_t;

            CallerAggregator &m_caller_aggregator;
            accessors_list_t const m_accessors_list;

            template <typename Accessor>
            using get_passed_argument_t = typename boost::mpl::at_c<PassedArguments, Accessor::index_t::value>::type;

          private:
            template <typename Accessor>
            using passed_argument_is_accessor_t = typename is_accessor<get_passed_argument_t<Accessor>>::type;

            template <typename Accessor>
            GT_FUNCTION constexpr auto get_passed_argument() const
                -> decltype(boost::fusion::at_c<Accessor::index_t::value>(m_accessors_list)) {
                return boost::fusion::at_c<Accessor::index_t::value>(m_accessors_list);
            }

          public:
            GT_FUNCTION constexpr function_aggregator_procedure_offsets(
                CallerAggregator &caller_aggregator, accessors_list_t const &list)
                : m_caller_aggregator(caller_aggregator), m_accessors_list(list) {}

            /**
             * @brief Accessor is a normal 3D accessor (not a global_accessor) and the passed Argument is an accessor
             * (not a local variable)
             */
            template <typename Accessor,
                enable_if_t<not is_global_accessor<Accessor>::value && passed_argument_is_accessor_t<Accessor>::value,
                    int> = 0>
            GT_FUNCTION constexpr auto operator()(Accessor const &accessor) const
                -> decltype(m_caller_aggregator(get_passed_argument_t<Accessor>())) {
                return m_caller_aggregator(get_passed_argument_t<Accessor>(
                    accessor_offset<0>(accessor) + Offi + accessor_offset<0>(get_passed_argument<Accessor>()),
                    accessor_offset<1>(accessor) + Offj + accessor_offset<1>(get_passed_argument<Accessor>()),
                    accessor_offset<2>(accessor) + Offk + accessor_offset<2>(get_passed_argument<Accessor>())));
            }

            /**
             * @brief Accessor is a global_accessor and the passed Argument is an accessor (not a local variable)
             */
            template <typename Accessor,
                enable_if_t<is_global_accessor<Accessor>::value && passed_argument_is_accessor_t<Accessor>::value,
                    int> = 0>
            GT_FUNCTION constexpr auto operator()(Accessor const &accessor) const
                -> decltype(m_caller_aggregator(get_passed_argument_t<Accessor>())) {
                return m_caller_aggregator(get_passed_argument<Accessor>());
            }

            /**
             * @brief Passed argument is a local variable (not an accessor)
             */
            template <typename Accessor>
            GT_FUNCTION constexpr typename std::enable_if<not passed_argument_is_accessor_t<Accessor>::value,
                typename boost::remove_reference<typename boost::fusion::result_of::at_c<accessors_list_t,
                    Accessor::index_t::value>::type>::type::type>::type &
            operator()(Accessor const &) const {
                return get_passed_argument<Accessor>().value();
            }

            template <class Op, class... Args>
            GT_FUNCTION constexpr auto operator()(expr<Op, Args...> const &arg) const
                GT_AUTO_RETURN(expressions::evaluation::value(*this, arg));
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

        \tparam Functor The stencil operator to be called
        \tparam Region The region in which to call it (to take the proper overload). A region with no exact match is
       not
       called and will result in compilation error. The user is responsible for calling the proper apply overload)
        \tparam Offi Offset along the i-direction (usually modified using at<...>)
        \tparam Offj Offset along the j-direction
        \tparam Offk Offset along the k-direction
    */
    template <typename Functor, typename Region = void, int Offi = 0, int Offj = 0, int Offk = 0>
    struct call_proc {

        GT_STATIC_ASSERT((is_interval<Region>::value or std::is_void<Region>::value),
            "Region should be a valid interval tag or void (default interval) to select the apply specialization in "
            "the called stencil function");

        /** This alias is used to move the computation at a certain offset
         */
        template <int I, int J, int K>
        using at = call_proc<Functor, Region, I, J, K>;

        /** With this interface a stencil function can be invoked and
            the offsets specified in the passed accessors are used to
            access values, w.r.t the offsets specified in a optional
            at<..> statement.
         */
        template <typename Evaluator, typename... Args>
        GT_FUNCTION static void with(Evaluator &eval, Args const &... args) {

            typedef _impl::function_aggregator_procedure_offsets<Evaluator,
                Offi,
                Offj,
                Offk,
                typename _impl::package_args<Args...>::type>
                f_aggregator_t;

            auto y = typename f_aggregator_t::accessors_list_t(_impl::make_wrap(args)...);

            auto agg_p = f_aggregator_t(eval, y);
            _impl::call_functor<Functor, Region>(agg_p);
        }
    };
} // namespace gridtools
