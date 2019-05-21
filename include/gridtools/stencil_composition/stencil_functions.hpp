/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#ifdef GT_ICOSAHEDRAL_GRIDS
#error Stencil functions are not suported for icosahedral grids
#endif

#include <type_traits>

#include "../common/hymap.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "accessor.hpp"
#include "dim.hpp"
#include "expressions/expr_base.hpp"
#include "interval.hpp"

namespace gridtools {
    // TODO: stencil functions work only for 3D stencils.

    namespace call_interfaces_impl_ {
        template <class Functor, class Region, class Eval>
        GT_FUNCTION enable_if_t<!std::is_void<Region>::value> call_functor(Eval &eval) {
            Functor::template apply<Eval &>(eval, Region{});
        }

        // overload for the default interval (Functor with one argument)
        template <class Functor, class Region, class Eval>
        GT_FUNCTION enable_if_t<std::is_void<Region>::value> call_functor(Eval &eval) {
            Functor::template apply<Eval &>(eval);
        }

        template <class Key>
        struct sum_offset_generator_f {
            using type = sum_offset_generator_f;

            using default_t = integral_constant<int_t, 0>;

            template <class Lhs, class Rhs>
            GT_FUNCTION auto operator()(Lhs &&lhs, Rhs &&rhs) const {
                return host_device::at_key_with_default<Key, default_t>(wstd::forward<Lhs>(lhs)) +
                       host_device::at_key_with_default<Key, default_t>(wstd::forward<Rhs>(rhs));
            }
        };

        template <class Res, class Lhs, class Rhs>
        GT_FUNCTION Res sum_offsets(Lhs &&lhs, Rhs &&rhs) {
            using keys_t = get_keys<Res>;
            using generators_t = meta::transform<sum_offset_generator_f, keys_t>;
            return tuple_util::host_device::generate<generators_t, Res>(
                wstd::forward<Lhs>(lhs), wstd::forward<Rhs>(rhs));
        }

        template <int_t I, int_t J, int_t K, class Accessor>
        GT_FUNCTION enable_if_t<I == 0 && J == 0 && K == 0, Accessor &&> get_offsets(Accessor &&acc) {
            return wstd::forward<Accessor>(acc);
        }

        template <int_t I,
            int_t J,
            int_t K,
            class Accessor,
            class Decayed = decay_t<Accessor>,
            class Size = tuple_util::size<Decayed>,
            class Res = array<int_t, Size::value>>
        GT_FUNCTION enable_if_t<I != 0 || J != 0 || K != 0, Res> get_offsets(Accessor &&acc) {
            static constexpr hymap::keys<dim::i, dim::j, dim::k>::
                values<integral_constant<int_t, I>, integral_constant<int_t, J>, integral_constant<int_t, K>>
                    offset = {};
            return sum_offsets<Res>(wstd::forward<Accessor>(acc), offset);
        }

        template <class Res, class Offsets>
        struct accessor_transform_f {
            GT_STATIC_ASSERT(is_accessor<Res>::value, GT_INTERNAL_ERROR);

            Offsets m_offsets;

            template <class Eval, class Src>
            GT_FUNCTION GT_CONSTEXPR decltype(auto) operator()(Eval &eval, Src &&src) const {
                return eval(sum_offsets<Res>(m_offsets, wstd::forward<Src>(src)));
            }
        };

        template <class Res, class Offsets>
        GT_FUNCTION accessor_transform_f<Res, Offsets> accessor_transform(Offsets &&offsets) {
            return {wstd::forward<Offsets>(offsets)};
        }

        template <class T>
        struct local_transform_f {
            T m_val;

            template <class Eval, class Src>
            GT_FUNCTION T operator()(Eval &&, Src &&) const {
                return m_val;
            }
        };

        template <int_t I, int_t J, int_t K>
        struct get_transform_f {
            template <class Accessor,
                class LazyParam,
                class Decayed = decay_t<Accessor>,
                class Param = typename LazyParam::type,
                enable_if_t<is_accessor<Decayed>::value &&
                                !(Param::intent_v == intent::inout && Decayed::intent_v == intent::in),
                    int> = 0>
            GT_FUNCTION auto operator()(Accessor &&accessor, LazyParam) const {
                return accessor_transform<Decayed>(get_offsets<I, J, K>(wstd::forward<Accessor>(accessor)));
            }

            template <class Arg,
                class Decayed = decay_t<Arg>,
                class LazyParam,
                class Param = typename LazyParam::type,
                enable_if_t<!is_accessor<Decayed>::value &&
                                !(Param::intent_v == intent::inout && std::is_const<remove_reference_t<Arg>>::value),
                    int> = 0>
            GT_FUNCTION GT_CONSTEXPR local_transform_f<Arg> operator()(Arg &&arg, LazyParam) const {
                return {wstd::forward<Arg>(arg)};
            }
        };

        template <class Eval, class Transforms>
        struct evaluator {
            Eval &m_eval;
            Transforms m_transforms;

            template <class Accessor,
                class Decayed = decay_t<Accessor>,
                enable_if_t<is_accessor<Decayed>::value, int> = 0>
            GT_FUNCTION decltype(auto) operator()(Accessor &&acc) const {
                return tuple_util::host_device::get<Decayed::index_t::value>(m_transforms)(
                    m_eval, wstd::forward<Accessor>(acc));
            }

            template <class Op, class... Ts>
            GT_FUNCTION auto operator()(expr<Op, Ts...> const &arg) const {
                return expressions::evaluation::value(*this, arg);
            }
        };
        template <class Eval, class Transforms>
        GT_FUNCTION evaluator<Eval, Transforms> make_evaluator(Eval &eval, Transforms &&transforms) {
            return {eval, wstd::forward<Transforms>(transforms)};
        }

        template <class Functor, class Region, int_t I, int_t J, int_t K, class Eval, class Args>
        GT_FUNCTION void evaluate_bound_functor(Eval &eval, Args &&args) {
            static constexpr meta::rename<meta::ctor<tuple<>>::apply,
                meta::transform<meta::defer<meta::id>::apply, typename Functor::param_list>>
                lazy_params = {};
            auto new_eval = make_evaluator(eval,
                tuple_util::host_device::transform(get_transform_f<I, J, K>{}, wstd::forward<Args>(args), lazy_params));
            call_functor<Functor, Region>(new_eval);
        }

        template <class Eval, class Arg, bool = is_accessor<Arg>::value>
        struct deduce_result_type : std::decay<decltype(std::declval<Eval &>()(std::declval<Arg &&>()))> {};

        template <class Eval, class Arg>
        struct deduce_result_type<Eval, Arg, false> : meta::lazy::id<Arg> {};

        /**
         * @brief Use forced return type (if not void) or deduce the return type.
         */
        template <class Eval, class ReturnType, class Arg, class...>
        struct get_result_type : std::conditional<std::is_void<ReturnType>::value,
                                     typename deduce_result_type<Eval, Arg>::type,
                                     ReturnType> {};

        template <class Accessor>
        using is_out_param = bool_constant<Accessor::intent_v == intent::inout>;
    } // namespace call_interfaces_impl_

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
    template <class Functor,
        class Region = void,
        class ReturnType = void,
        int_t OffI = 0,
        int_t OffJ = 0,
        int_t OffK = 0>
    class call {
        GT_STATIC_ASSERT(is_interval<Region>::value or std::is_void<Region>::value,
            "Region should be a valid interval tag or void (default interval) to select the apply specialization in "
            "the called stencil function");

        using params_t = typename Functor::param_list;
        using out_params_t = meta::filter<call_interfaces_impl_::is_out_param, params_t>;

        GT_STATIC_ASSERT(meta::length<out_params_t>::value == 1,
            "Trying to invoke stencil operator with more than one output as a function");

        using out_param_t = meta::first<out_params_t>;
        static constexpr size_t out_param_index = out_param_t::index_t::value;

      public:
        /** This alias is used to move the computation at a certain offset
         */
        template <int_t I, int_t J, int_t K>
        using at = call<Functor, Region, ReturnType, I, J, K>;

        /**
         * @brief alias to set the return type, e.g.
         */
        template <typename ForcedReturnType>
        using return_type = call<Functor, Region, ForcedReturnType, OffI, OffJ, OffK>;

        /**
         * With this interface a stencil function can be invoked and the offsets specified in the passed accessors are
         * used to access values, w.r.t the offsets specified in a optional  at<..> statement.
         */
        template <class Eval,
            class... Args,
            class Res = typename call_interfaces_impl_::get_result_type<Eval, ReturnType, decay_t<Args>...>::type,
            enable_if_t<sizeof...(Args) + 1 == meta::length<params_t>::value, int> = 0>
        GT_FUNCTION static Res with(Eval &eval, Args &&... args) {
            Res res;
            call_interfaces_impl_::evaluate_bound_functor<Functor, Region, OffI, OffJ, OffK>(eval,
                tuple_util::host_device::insert<out_param_index>(res, tuple<Args &&...>{wstd::forward<Args>(args)...}));
            return res;
        }
    };

    /**
     * Main interface for calling stencil operators as functions with side-effects. The interface accepts a list of
     * arguments to be passed to the called function and these arguments can be accessors or simple values.
     *
     * Usage : call_proc<functor, region>::[at<offseti, offsetj, offsetk>::]with(eval, accessors_or_values...);
     *
     * Accessors_or_values referes to a list of arguments that may be accessors of the caller functions or local
     * variables of the type accessed (or converted to) by the accessor in the called function, where the results should
     * be obtained from. The values can also be used by the function as inputs.
     *
     * \tparam Functor The stencil operator to be called
     * \tparam Region The region in which to call it (to take the proper overload). A region with no exact match is not
     *    called and will result in compilation error. The user is responsible for calling the proper apply overload)
     * \tparam OffI Offset along the i-direction (usually modified using at<...>)
     * \tparam OffJ Offset along the j-direction
     * \tparam OffK Offset along the k-direction
     * */
    template <class Functor, class Region = void, int_t OffI = 0, int_t OffJ = 0, int_t OffK = 0>
    struct call_proc {

        GT_STATIC_ASSERT((is_interval<Region>::value or std::is_void<Region>::value),
            "Region should be a valid interval tag or void (default interval) to select the apply specialization in "
            "the called stencil function");

        /** This alias is used to move the computation at a certain offset
         */
        template <int_t I, int_t J, int_t K>
        using at = call_proc<Functor, Region, I, J, K>;

        /**
         * With this interface a stencil function can be invoked and the offsets specified in the passed accessors are
         * used to access values, w.r.t the offsets specified in a optional at<..> statement.
         */
        template <class Eval, class... Args>
        GT_FUNCTION static enable_if_t<sizeof...(Args) == meta::length<typename Functor::param_list>::value> with(
            Eval &eval, Args &&... args) {
            call_interfaces_impl_::evaluate_bound_functor<Functor, Region, OffI, OffJ, OffK>(
                eval, tuple<Args &&...>{wstd::forward<Args>(args)...});
        }
    };
} // namespace gridtools
