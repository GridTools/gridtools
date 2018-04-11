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

/**
 *  @file
 *
 *  Here is a set of algorithms that are defined on "tuple like" structures
 *  To be a "tuple like", a structure should:
 *    - be a template instantiation of a template of class parameters [ for ex. foo<int, double, int> ]
 *    - have accessors like do_get(std::integral_constant< size_t, I >, foo<Ts...>) defined in
 *      gridtools::tuple_util::traits namespace, or being available via ADL
 *    - have an element wise ctor
 *    - have a ctor from another tuple of the same kind. [ i.e. foo<double, double> should be
 *      constructible from foo<int, int> or foo<double&&, double&&>]
 *
 *  If the opposite is not mentioned explicitly, the algorithms produce tuples of references. L-value or R-value
 *  depending on algorithm input.
 *
 *  Almost all algorithms are defined in two forms:
 *    1) conventional template functions;
 *    2) functions that return generic functors
 *
 *  For example you can do:
 *    auto ints = transform([](int x) {return x;}, input);
 *  our you can:
 *    auto convert_to_ints = transform([](int x) {return x;});
 *    auto ints = convert_to_ints(input);
 *
 *  The second form is more composable. For example if the input is a tuple of tuples of whatever and you need a tuple
 *  of tuple of tuple of integers you can do it in one expression:
 *  auto out = transform(transform([](int x) {return x;}), input);
 *
 *
 *  TODO list
 *  =========
 *  - extend concept to be applied to std::array's
 *  - add for_each_in_cartesian_product
 *  - adapt gridtools::array, gridtools::pair and gridtools::tuple
 *  - supply all functions here with `GT_FUNCTION` variants.
 *  - add apply (generic version of std::apply)
 *  - add push_front
 *  - add for_each_index
 *
 */
#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include "defs.hpp"
#include "functional.hpp"

#include "generic_metafunctions/type_traits.hpp"
#include "generic_metafunctions/meta.hpp"

namespace gridtools {

    namespace tuple_util {

        namespace traits {

            /// std::tuple adaptation
            template < size_t I, class... Ts >
            constexpr typename std::tuple_element< I, std::tuple< Ts... > >::type &do_get(
                std::integral_constant< size_t, I >, std::tuple< Ts... > &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class... Ts >
            constexpr typename std::tuple_element< I, std::tuple< Ts... > >::type const &do_get(
                std::integral_constant< size_t, I >, std::tuple< Ts... > const &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class... Ts >
            constexpr typename std::tuple_element< I, std::tuple< Ts... > >::type &&do_get(
                std::integral_constant< size_t, I >, std::tuple< Ts... > &&obj) noexcept {
                return std::get< I >(std::move(obj));
            }

            /// std::array adaptation
            template < size_t I, class T, size_t N >
            constexpr T &do_get(std::integral_constant< size_t, I >, std::array< T, N > &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class T, size_t N >
            constexpr T const &do_get(std::integral_constant< size_t, I >, std::array< T, N > const &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class T, size_t N >
            constexpr T &&do_get(std::integral_constant< size_t, I >, std::array< T, N > &&obj) noexcept {
                return std::get< I >(std::move(obj));
            }

            /// std::pair adaptation
            template < size_t I, class T1, class T2 >
            constexpr typename std::tuple_element< I, std::pair< T1, T2 > >::type &do_get(
                std::integral_constant< size_t, I >, std::pair< T1, T2 > &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class T1, class T2 >
            constexpr const typename std::tuple_element< I, std::pair< T1, T2 > >::type &do_get(
                std::integral_constant< size_t, I >, const std::pair< T1, T2 > &obj) noexcept {
                return std::get< I >(obj);
            }
            template < size_t I, class T1, class T2 >
            constexpr typename std::tuple_element< I, std::pair< T1, T2 > >::type &&do_get(
                std::integral_constant< size_t, I >, std::pair< T1, T2 > &&obj) noexcept {
                return std::get< I >(std::move(obj));
            }

            template < size_t I >
            struct get_f {
                template < class T >
                constexpr auto operator()(T &&obj) const
                    GT_AUTO_RETURN(do_get(std::integral_constant< size_t, I >{}, std::forward< T >(obj)));
            };
        }

        /// This is tuple element accessor. Like std::get, but extendable via defining do_get
        using traits::get_f;
        template < size_t I, class T >
        constexpr auto get(T &&obj) noexcept GT_AUTO_RETURN(get_f< I >{}(std::forward< T >(obj)));

        namespace _impl {

            template < class GeneratorList, class Res >
            struct generate_f;
            template < template < class... > class L, class... Generators, class Res >
            struct generate_f< L< Generators... >, Res > {
                template < class... Args >
                Res operator()(Args &&... args) const {
                    return Res{Generators{}(std::forward< Args >(args)...)...};
                }
            };

            enum class ref_kind { rvalue, lvalue, const_lvalue };

            template < class >
            struct get_ref_kind;

            template < class T >
            struct get_ref_kind< T && > : std::integral_constant< ref_kind, ref_kind::rvalue > {};

            template < class T >
            struct get_ref_kind< T & > : std::integral_constant< ref_kind, ref_kind::lvalue > {};

            template < class T >
            struct get_ref_kind< T const & > : std::integral_constant< ref_kind, ref_kind::const_lvalue > {};

            template < ref_kind Kind, class Dst >
            struct add_ref;

            template < class T >
            struct add_ref< ref_kind::rvalue, T > : std::add_rvalue_reference< T > {};

            template < class T >
            struct add_ref< ref_kind::lvalue, T > : std::add_lvalue_reference< T > {};

            template < class T >
            struct add_ref< ref_kind::const_lvalue, T > : std::add_lvalue_reference< add_const_t< T > > {};

            template < size_t I >
            struct transform_elem_f {
                template < class Fun, class... Tups >
                auto operator()(Fun &&fun, Tups &&... tups) const
                    GT_AUTO_RETURN(std::forward< Fun >(fun)(get< I >(std::forward< Tups >(tups)...)));
            };

#if GT_BROKEN_TEMPLATE_ALIASES
            template < ref_kind Kind >
            struct get_accessor {
                template < class T >
                struct apply : add_ref< Kind, T > {};
            };

            template < class Fun >
            struct get_fun_result {
                template < class... Ts >
                struct apply {
                    using type = decltype(std::declval< Fun >()(std::declval< Ts >()...));
                };
            };

            template < class I >
            struct get_transform_generator {
                using type = transform_elem_f< I::value >;
            };

#else
            template < ref_kind Kind >
            struct get_accessor {
                template < class T >
                using apply = typename add_ref< Kind, T >::type;
            };

            template < class Fun >
            struct get_fun_result {
                template < class... Ts >
                using apply = decltype(std::declval< Fun >()(std::declval< Ts >()...));
            };

            template < class I >
            using get_transform_generator = transform_elem_f< I::value >;
#endif

            template < class Tup >
            GT_META_DEFINE_ALIAS(get_accessors,
                meta::transform,
                (get_accessor< get_ref_kind< Tup >::value >::template apply, decay_t< Tup >));

            template < class Fun >
            struct transform_f {
                template < class... Args >
                GT_META_DEFINE_ALIAS(get_results_t, meta::transform, (get_fun_result< Fun >::template apply, Args...));

                Fun m_fun;

                template < class Tup,
                    class... Tups,
                    class Res = GT_META_CALL(
                        get_results_t, (GT_META_CALL(get_accessors, Tup &&), GT_META_CALL(get_accessors, Tups &&)...)) >
                Res operator()(Tup &&tup, Tups &&... tups) const {
                    constexpr auto length = meta::length< decay_t< Tup > >::value;
                    using generators = GT_META_CALL(
                        meta::transform, (get_transform_generator, GT_META_CALL(meta::make_indices, length)));
                    return generate_f< generators, Res >{}(
                        m_fun, std::forward< Tup >(tup), std::forward< Tups >(tups)...);
                }
            };
            template < class >
            struct for_each_impl_f;

            template < template < class T, T... > class L, class Int, Int... Is >
            struct for_each_impl_f< L< Int, Is... > > {
                template < class Fun, class... Tups >
                void operator()(Fun &&fun, Tups &&... tups) const {
                    void((int[]){(std::forward< Fun >(fun)(get< Is >(std::forward< Tups >(tups)...)), 0)...});
                }
            };

            template < class Fun >
            struct for_each_f {
                Fun m_fun;

                template < class Tup, class... Tups >
                void operator()(Tup &&tup, Tups &&... tups) const {
                    for_each_impl_f< make_gt_index_sequence< meta::length< decay_t< Tup > >::value > >{}(
                        m_fun, std::forward< Tup >(tup), std::forward< Tups >(tups)...);
                }
            };

            struct flatten_f {
                template < size_t OuterI, size_t InnerI >
                struct generator_f {
                    template < class Tup >
                    auto operator()(Tup &&tup) const
                        GT_AUTO_RETURN(get< InnerI >(get< OuterI >(std::forward< Tup >(tup))));
                };

#if GT_BROKEN_TEMPLATE_ALIASES
                template < class OuterI, class InnerI >
                struct get_generator {
                    using type = generator_f< OuterI::value, InnerI::value >;
                };
#else
                template < class OuterI, class InnerI >
                using get_generator = generator_f< OuterI::value, InnerI::value >;
#endif

                template < class OuterI, class InnerTup >
                GT_META_DEFINE_ALIAS(get_inner_generators,
                    meta::transform,
                    (meta::bind< get_generator, OuterI, meta::_1 >::template apply,
                                         GT_META_CALL(meta::make_indices_for, InnerTup)));

                template < class Tup,
                    class Accessors = GT_META_CALL(
                        meta::transform, (get_accessors, GT_META_CALL(get_accessors, Tup &&))),
                    class Res = GT_META_CALL(meta::flatten, Accessors) >
                Res operator()(Tup &&tup) const {
                    using generators = GT_META_CALL(meta::flatten,
                        (GT_META_CALL(meta::transform,
                            (get_inner_generators, GT_META_CALL(meta::make_indices_for, Accessors), Accessors))));
                    return generate_f< generators, Res >{}(std::forward< Tup >(tup));
                }
            };

            template < size_t N >
            struct drop_front_f {
#if GT_BROKEN_TEMPLATE_ALIASES
                template < class I >
                struct get_drop_front_generator {
                    using type = get_f< N + I::value >;
                };
#else
                template < class I >
                using get_drop_front_generator = get_f< N + I::value >;
#endif

                template < class Tup,
                    class Accessors = GT_META_CALL(get_accessors, Tup &&),
                    class Res = GT_META_CALL(meta::drop_front_c, (N, Accessors)) >
                Res operator()(Tup &&tup) const {
                    using generators =
                        GT_META_CALL(meta::transform,
                            (get_drop_front_generator,
                                         GT_META_CALL(meta::make_indices, meta::length< Accessors >::value - N)));
                    return generate_f< generators, Res >{}(std::forward< Tup >(tup));
                }
            };

            template < class, class >
            struct push_back_impl_f;

            template < template < class T, T... > class L, class Int, Int... Is, class Res >
            struct push_back_impl_f< L< Int, Is... >, Res > {
                template < class Tup, class... Args >
                Res operator()(Tup &&tup, Args &&... args) const {
                    return Res{get< Is >(std::forward< Tup >(tup))..., std::forward< Args >(args)...};
                }
            };

            struct push_back_f {
                template < class Tup,
                    class... Args,
                    class Accessors = GT_META_CALL(get_accessors, Tup &&),
                    class Res = GT_META_CALL(meta::push_back, (Accessors, Args &&...)) >
                Res operator()(Tup &&tup, Args &&... args) const {
                    return push_back_impl_f< make_gt_index_sequence< meta::length< Accessors >::value >, Res >{}(
                        std::forward< Tup >(tup), std::forward< Args >(args)...);
                }
            };

            template < class Fun >
            struct fold_f {
#if GT_BROKEN_TEMPLATE_ALIASES
                template < class S, class T >
                struct meta_fun : get_fun_result< Fun >::template apply< S, T > {};
#else
                template < class S, class T >
                using meta_fun = typename get_fun_result< Fun >::template apply< S, T >;
#endif

                Fun m_fun;

                template < class State, template < class... > class L >
                State operator()(State &&state, L<>) const {
                    return state;
                }

                template < class State,
                    class Tup,
                    class Accessors = GT_META_CALL(get_accessors, Tup &&),
                    class Res = GT_META_CALL(meta::lfold, (meta_fun, State &&, Accessors)) >
                Res operator()(State &&state, Tup &&tup) const {
                    auto &&new_state = m_fun(std::forward< State >(state), get< 0 >(std::forward< Tup >(tup)));
                    auto &&rest = drop_front_f< 1 >{}(std::forward< Tup >(tup));
                    return this->operator()(std::move(new_state), std::move(rest));
                }

                template < class Tup >
                auto operator()(Tup &&tup) const GT_AUTO_RETURN(this->operator()(
                    get< 0 >(std::forward< Tup >(tup)), drop_front_f< 1 >{}(std::forward< Tup >(tup))));
            };
        }

        /// like boost::fusion::transform, but not lazy and can take any number of tuples as input.
        ///  in case of multiple inputs all inputs have the same size obviously.
        template < class Fun, class Tup, class... Tups >
        auto transform(Fun &&fun, Tup &&tup, Tups &&... tups) GT_AUTO_RETURN(_impl::transform_f< Fun >{
            std::forward< Fun >(fun)}(std::forward< Tup >(tup), std::forward< Tups >(tups)...));

        template < class Fun >
        constexpr _impl::transform_f< Fun > transform(Fun fun) {
            return {std::move(fun)};
        }

        /// like boost::fusion::for_each, but can take any number of tuples as input
        ///
        template < class Fun, class Tup, class... Tups >
        void for_each(Fun &&fun, Tup &&tup, Tups &&... tups) {
            _impl::for_each_f< Fun >{std::forward< Fun >(fun)}(std::forward< Tup >(tup), std::forward< Tups >(tups)...);
        }

        template < class Fun >
        constexpr _impl::for_each_f< Fun > for_each(Fun fun) {
            return {std::move(fun)};
        }

        inline constexpr _impl::flatten_f flatten() { return {}; }

        template < class Tup >
        auto flatten(Tup &&tup) GT_AUTO_RETURN(flatten()(std::forward< Tup >(tup)));

        /// Generators is a typelist of functors. Elements in Generators typelist are default constructed.
        /// Then they are invoked with provided arguments each. The results are passed to the constructor of Res.
        /// The created object is returned.
        template < class Generators, class Res, class... Args >
        Res generate(Args &&... args) {
            return _impl::generate_f< Generators, Res >{}(std::forward< Args >(args)...);
        }

        template < size_t N >
        constexpr _impl::drop_front_f< N > drop_front() {
            return {};
        }

        template < size_t N, class Tup >
        auto drop_front(Tup &&tup) GT_AUTO_RETURN(drop_front< N >()(std::forward< Tup >(tup)));

        inline constexpr _impl::push_back_f push_back() { return {}; }

        template < class Tup, class... Args >
        auto push_back(Tup &&tup, Args &&... args)
            GT_AUTO_RETURN(push_back()(std::forward< Tup >(tup), std::forward< Args >(args)...));

        /// Left fold.
        /// If there are tree parameters, the first is a binary function to fold with,
        /// second is an initial state, and the third is a tuple to fold.
        /// In the case if there is only two parameters, the second parameter is a (non-empty) tuple to fold and
        /// it it's first element acts as an initial state.
        template < class Fun, class Arg, class... Args >
        auto fold(Fun &&fun, Arg &&arg, Args &&... args) GT_AUTO_RETURN(
            _impl::fold_f< Fun >{std::forward< Fun >(fun)}(std::forward< Arg >(arg), std::forward< Args >(args)...));

        template < class Fun >
        constexpr _impl::fold_f< Fun > fold(Fun fun) {
            return {std::move(fun)};
        }

        inline constexpr _impl::transform_f< clone > deep_copy() { return {}; }

        /// All the references within input are copied as values into output. Output type doesn't contain references.
        ///
        template < class Tup >
        auto deep_copy(Tup &&tup) GT_AUTO_RETURN(deep_copy()(std::forward< Tup >(tup)));
    }
}
