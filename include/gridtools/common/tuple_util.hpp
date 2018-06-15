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

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include "defs.hpp"
#include "functional.hpp"

#include "generic_metafunctions/type_traits.hpp"
#include "generic_metafunctions/meta.hpp"

namespace gridtools {

    /** \ingroup common
        @{
    */
    /** \defgroup tupleutils Utilities for Tuples
        @{
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
     *  The second form is more composable. For example if the input is a tuple of tuples of whatever and you need a
     *  tuple of tuple of tuple of integers you can do it in one expression:
     *  auto out = transform(transform([](int x) {return x;}), input);
     *
     *
     *  TODO list
     *  =========
     *  - extend concept to be applied to std::array's
     *  - adapt gridtools::array, gridtools::pair and gridtools::tuple
     *  - supply all functions here with `GT_FUNCTION` variants.
     *  - add apply (generic version of std::apply)
     *  - add push_front
     *  - add for_each_index
     *  - add filter
     *
     */
    namespace tuple_util {

        /// @cond
        namespace traits {

            /*
             * Adaptation traits
             *
             * To enable the algorithms provided in this file for a tuple-like container, an implementation of do_get
             * has to be defined here. Currently std::tuple and std::pair are fully supported. Limited support is also
             * provided for std::array (limitation due to the different template parameters).
             */

            // std::tuple adaptation
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

            // std::array adaptation
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

            // std::pair adaptation
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
        /// @endcond

        using traits::get_f;

        /**
         * @brief Tuple element accessor like std::get.
         *
         * @tparam I Element index.
         * @tparam T Tuple-like type.
         * @param obj Tuple-like object.
         *
         * Extensible via defining `do_get()`-functions which are searched via argument-dependent lookup.
         */
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
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 9
                // CAUTION!! CUDA8 barely understands this. If you just GT_AUTO_RETURN it goes nuts with mysterious
                // error message; if you replace the inner result_of_t to typename std::result_of<...>::type it fails
                // as well.
                // Alternatively you can also write:
                // auto operator()(Fun &&fun, Tups &&... tups) const
                // -> typename std::result_of<Fun&&(decltype(get< I >(std::forward< Tups >(tups)))...)>::type
                template < class Fun, class... Tups >
                typename std::result_of< Fun && (result_of_t< get_f< I >(Tups &&) >...) >::type operator()(
                    Fun &&fun, Tups &&... tups) const {
                    return std::forward< Fun >(fun)(get< I >(std::forward< Tups >(tups))...);
                }
#elif(defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1800) || \
    (defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 10)
                template < class Fun, class Tup >
                auto operator()(Fun &&fun, Tup &&tup) const
                    GT_AUTO_RETURN(std::forward< Fun >(fun)(get< I >(std::forward< Tup >(tup))));
                template < class Fun, class Tup1, class Tup2 >
                auto operator()(Fun &&fun, Tup1 &&tup1, Tup2 &&tup2) const GT_AUTO_RETURN(std::forward< Fun >(fun)(
                    get< I >(std::forward< Tup1 >(tup1)), get< I >(std::forward< Tup2 >(tup2))));
                template < class Fun, class Tup1, class Tup2, class Tup3 >
                auto operator()(Fun &&fun, Tup1 &&tup1, Tup2 &&tup2, Tup3 &&tup3) const
                    GT_AUTO_RETURN(std::forward< Fun >(fun)(get< I >(std::forward< Tup1 >(tup1)),
                        get< I >(std::forward< Tup2 >(tup2)),
                        get< I >(std::forward< Tup3 >(tup3))));
#else
                template < class Fun, class... Tups >
                auto operator()(Fun &&fun, Tups &&... tups) const
                    GT_AUTO_RETURN(std::forward< Fun >(fun)(get< I >(std::forward< Tups >(tups))...));
#endif
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
                        meta::transform, (get_transform_generator, GT_META_CALL(meta::make_indices_c, length)));
                    return generate_f< generators, Res >{}(
                        m_fun, std::forward< Tup >(tup), std::forward< Tups >(tups)...);
                }
            };

            struct empty {};

            template < class Fun >
            struct for_each_adaptor_f {
                Fun m_fun;
                template < class... Args >
                empty operator()(Args &&... args) const {
                    m_fun(std::forward< Args >(args)...);
                    return {};
                }
            };

            template < class Indices >
            struct apply_to_elements_f;

            template < template < class... > class L, class... Is >
            struct apply_to_elements_f< L< Is... > > {
                template < class Fun, class... Tups >
                auto operator()(Fun &&fun, Tups &&... tups) const
                    GT_AUTO_RETURN(std::forward< Fun >(fun)(get< Is::value >(std::forward< Tups >(tups))...));
            };

            template < class >
            struct for_each_in_cartesian_product_impl_f;

            template < template < class... > class Outer, class... Inners >
            struct for_each_in_cartesian_product_impl_f< Outer< Inners... > > {
                template < class Fun, class... Tups >
                void operator()(Fun &&fun, Tups &&... tups) const {
                    void((int[]){
                        (apply_to_elements_f< Inners >{}(std::forward< Fun >(fun), std::forward< Tups >(tups)...),
                            0)...});
                }
            };

            template < class Fun >
            struct for_each_in_cartesian_product_f {
                Fun m_fun;
                template < class... Tups >
                void operator()(Tups &&... tups) const {
                    for_each_in_cartesian_product_impl_f< GT_META_CALL(
                        meta::cartesian_product, (GT_META_CALL(meta::make_indices_for, decay_t< Tups >)...)) >{}(
                        m_fun, std::forward< Tups >(tups)...);
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
                                         GT_META_CALL(meta::make_indices_c, meta::length< Accessors >::value - N)));
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

        /**
         * @brief Transforms each tuple element by a function.
         *
         * Transformations with functions with more than one argument are supported by passing multiple tuples of the
         * same size.
         *
         * @tparam Fun Functor type.
         * @tparam Tup Optional tuple-like type.
         * @tparam Tups Optional Tuple-like types.
         *
         * @param fun Function that should be applied to all elements of the given tuple(s).
         * @param tup First tuple-like object, serves as first arguments to `fun` if given.
         * @param tups Further tuple-like objects, serve as additional arguments to `fun` if given.
         *
         * Example code:
         * @code
         * #include <functional>
         * using namespace std::placeholders;
         *
         * struct add {
         *     template < class A, class B >
         *     auto operator()(A a, B b) -> decltype(a + b) {
         *         return a + b;
         *     }
         * };
         *
         * // Unary function, like boost::fusion::transform
         * auto tup = std::make_tuple(1, 2, 3.5);
         * auto fun = std::bind(add{}, 2, _1);
         * auto res = transform(fun, tup);
         * // res == {3, 4, 5.5}
         *
         * // Binary function
         * auto tup2 = std::make_tuple(1.5, 3, 4.1);
         * auto res2 = transform(add{}, tup, tup2);
         * // res2 == {2.5, 5, 7.6}
         * @endcode
         */
        template < class Fun, class Tup, class... Tups >
        auto transform(Fun &&fun, Tup &&tup, Tups &&... tups) GT_AUTO_RETURN(_impl::transform_f< Fun >{
            std::forward< Fun >(fun)}(std::forward< Tup >(tup), std::forward< Tups >(tups)...));

        /**
         * @brief Returns a functor that transforms each tuple element by a function.
         *
         * Composable version of `transform` that returns a functor which can be invoked with (one or multiple) tuples.
         *
         * @tparam Fun Functor type.
         *
         * @param fun Function that should be applied to all elements of the given tuple(s).
         *
         * Example code:
         * @code
         * struct add {
         *     template < class A, class B >
         *     auto operator()(A a, B b) -> decltype(a + b) {
         *         return a + b;
         *     }
         * };
         *
         * // Composable usage with only a function argument
         * auto addtuples = transform(add{});
         * // addtuples takes now two tuples as arguments
         * auto tup1 = std::make_tuple(1, 2, 3.5);
         * auto tup2 = std::make_tuple(1.5, 3, 4.1);
         * auto res = addtuples(tup1, tup2)
         * // res == {2.5, 5, 7.6}
         * @endcode
         */
        template < class Fun >
        constexpr _impl::transform_f< Fun > transform(Fun fun) {
            return {std::move(fun)};
        }

        /**
         * @brief Calls a function for each element in a tuple.
         *
         * Functions with more than one argument are supported by passing multiple tuples of the same size. If only a
         * function but no tuples are passed, a composable functor is returned.
         *
         * @tparam Fun Functor type.
         * @tparam Tup Optional tuple-like type.
         * @tparam Tups Optional Tuple-like types.
         *
         * @param fun Function that should be called for each element of the given tuple(s).
         * @param tup First tuple-like object, serves as first arguments to `fun` if given.
         * @param tups Further tuple-like objects, serve as additional arguments to `fun` if given.
         *
         * Example code:
         * @code
         * struct sum {
         *     double& value;
         *     template < class A >
         *     void operator()(A a, bool mask = true) const {
         *         if (mask)
         *             value += a;
         *     }
         * };
         *
         * // Unary function, like boost::fusion::for_each
         * auto tup = std::make_tuple(1, 2, 3.5);
         * double sum_value = 0.0;
         * for_each(sum{sum_value}, tup);
         * // sum_value == 6.5
         *
         * // Binary function
         * auto tup2 = std::make_tuple(false, true, true);
         * sum_value = 0.0;
         * for_each(sum{sum_value}, tup, tup2);
         * // sum_value == 5.5
         * @endcode
         */
        template < class Fun, class Tup, class... Tups >
        void for_each(Fun &&fun, Tup &&tup, Tups &&... tups) {
            transform(_impl::for_each_adaptor_f< Fun >{std::forward< Fun >(fun)},
                std::forward< Tup >(tup),
                std::forward< Tups >(tups)...);
        }

        /**
         * @brief Returns a functor that calls a function for each element in a tuple.
         *
         * Composable version of `for_each` that returns a functor which can be invoked with (one or multiple) tuples.
         *
         * @tparam Fun Functor type.
         *
         * @param fun Function that should be called for each element of the given tuple(s).
         *
         * Example code:
         * @code
         * struct sum {
         *     double& value;
         *     template < class A >
         *     void operator()(A a, bool mask = true) const {
         *         if (mask)
         *             value += a;
         *     }
         * };
         *
         * // Composable usage with only a function argument
         * sum_value = 0.0;
         * auto sumtuples = for_each(sum{sum_value});
         * // sumtuples takes now two tuples as arguments
         * auto tup1 = std::make_tuple(1, 2, 3.5);
         * auto tup2 = std::make_tuple(false, true, true);
         * auto res = sumtuples(tup1, tup2)
         * // sum_value == 5.5
         * @endcode
         */
        template < class Fun >
        constexpr _impl::transform_f< _impl::for_each_adaptor_f< Fun > > for_each(Fun fun) {
            return {{std::move(fun)}};
        }

        /**
         * @brief Calls a function for each element in a cartesian product of the given tuples.
         *
         * If only a function but no tuples are passed, a composable functor is returned.
         *
         * @tparam Fun Functor type.
         * @tparam Tup Optional tuple-like type.
         * @tparam Tups Optional Tuple-like types.
         *
         * @param fun Function that should be called for each element in a cartesian product of the given tuples.
         * @param tup First tuple-like object, serves as first arguments to `fun` if given.
         * @param tups Further tuple-like objects, serve as additional arguments to `fun` if given.
         *
         * Example code:
         * @code
         * struct sum {
         *     double& value;
         *     template < class A, class B >
         *     void operator()(A a, B b) const {
         *         if (mask)
         *             value += a * b;
         *     }
         * };
         *
         * // Binary function
         * sum_value = 0.;
         * for_each(sum{sum_value}, std::make_tuple(1, 2, 3), std::make_tuple(1, 10));
         * // sum_value == 66.
         * @endcode
         */
        template < class Fun, class Tup, class... Tups >
        void for_each_in_cartesian_product(Fun &&fun, Tup &&tup, Tups &&... tups) {
            _impl::for_each_in_cartesian_product_f< Fun >{std::forward< Fun >(fun)}(
                std::forward< Tup >(tup), std::forward< Tups >(tups)...);
        }

        template < class Fun >
        constexpr _impl::for_each_in_cartesian_product_f< Fun > for_each_in_cartesian_product(Fun fun) {
            return {std::move(fun)};
        }

        /**
         * @brief Return a functor that flattens a tuple of tuples non-recursively into a single tuple.
         *
         * Flattens only the first two levels of nested tuples into a single level. Does not flatten further levels of
         * nesting.
         *
         * Example:
         * @code
         * auto flattenfunc = flatten();
         * auto tup1 = std::make_tuple(1, 2);
         * auto tup2 = std::make_tuple(3, 4, 5);
         * auto flat = flattenfunc(tup1, tup2);
         * // flat == {1, 2, 3, 4, 5}
         * @endcode
         */
        inline constexpr _impl::flatten_f flatten() { return {}; }

        /**
         * @brief Non-recursively flattens a tuple of tuples into a single tuple.
         *
         * Flattens only the first two levels of nested tuples into a single level. Does not flatten further levels of
         * nesting.
         *
         * @tparam Tup Tuple-like type.
         * @param tup Tuple-like object.
         *
         * Example:
         * @code
         * auto tup1 = std::make_tuple(1, 2);
         * auto tup2 = std::make_tuple(3, 4, 5);
         * auto flat = flatten(tup1, tup2);
         * // flat == {1, 2, 3, 4, 5}
         * @endcode
         */
        template < class Tup >
        auto flatten(Tup &&tup) GT_AUTO_RETURN(flatten()(std::forward< Tup >(tup)));

        /**
         * @brief Constructs an object from generator functors.
         *
         * `Generators` is a typelist of generator functors. Instances of those types are first default constructed,
         * then invoked with `args` as arguments. The results of those calls are then passed to the constructor of
         * `Res`.
         *
         * @tparam Generators A typelist of functors. All functor types must be default-constructible and callable with
         * arguments of type `Args`.
         * @tparam Res The type that should be constructed.
         * @tparam Args Argument types for the generator functors.
         *
         * @param args Arguments that will be passed to the generator functors.
         *
         * Example:
         * @code
         * // Generators that extract some information from the given arguments (a single std::string in this example)
         * struct ptr_extractor {
         *     const char* operator()(std::string const& s) const {
         *         return s.data();
         *     }
         * };
         *
         * struct size_extractor {
         *     std::size_t operator()(std::string const& s) const {
         *         return s.size();
         *     }
         * };
         *
         * // We want to generate a pair of a pointer and size, that represents this string in a simple C-style manner
         * std::string s = "Hello World!";
         * // Target-type to construct
         * using ptr_size_pair = std::pair< const char*, std::size_t >;
         *
         * // Typelist of generators
         * using generators = std::tuple< ptr_extractor, size_extractor>;
         *
         * // Generate pair
         * auto p = generate< generators, ptr_size_pair >(s);
         * // p.first is now a pointer to the first character of s, p.second is the size of s
         * @endcode
         */
        template < class Generators, class Res, class... Args >
        Res generate(Args &&... args) {
            return _impl::generate_f< Generators, Res >{}(std::forward< Args >(args)...);
        }

        /**
         * @brief Returns a functor that removes the first `N` elements from a tuple.
         *
         * @tparam N Number of elements to remove.
         *
         * Example:
         * @code
         * auto dropper = drop_front<2>();
         * auto tup = std::make_tuple(1, 2, 3, 4);
         * auto res = dropper(tup);
         * // res == {3, 4}
         * @endcode
         */
        template < size_t N >
        constexpr _impl::drop_front_f< N > drop_front() {
            return {};
        }

        /**
         * @brief Removes the first `N` elements from a tuple.
         *
         * @tparam N Number of elements to remove.
         * @tparam Tup Tuple-like type.
         *
         * @param tup Tuple to remove first `N` elements from.
         *
         * Example:
         * @code
         * auto tup = std::make_tuple(1, 2, 3, 4);
         * auto res = drop_front<2>(tup);
         * // res == {3, 4}
         * @endcode
         */
        template < size_t N, class Tup >
        auto drop_front(Tup &&tup) GT_AUTO_RETURN(drop_front< N >()(std::forward< Tup >(tup)));

        /**
         * @brief Returns a functor that appends elements to a tuple.
         *
         * Example:
         * @code
         * auto pusher = push_back();
         * auto tup = std::make_tuple(1, 2);
         * auto res = pusher(tup, 3, 4);
         * // res = {1, 2, 3, 4}
         * @endcode
         */
        inline constexpr _impl::push_back_f push_back() { return {}; }

        /**
         * @brief Appends elements to a tuple.
         *
         * @tparam Tup Tuple-like type.
         * @tparam Args Argument types to append.
         *
         * @param tup Tuple-like object.
         * @param args Arguments to append.
         *
         * Example:
         * @code
         * auto tup = std::make_tuple(1, 2);
         * auto res = push_back(tup, 3, 4);
         * // res = {1, 2, 3, 4}
         * @endcode
         */
        template < class Tup, class... Args >
        auto push_back(Tup &&tup, Args &&... args)
            GT_AUTO_RETURN(push_back()(std::forward< Tup >(tup), std::forward< Args >(args)...));

        /**
         * @brief Left fold on tuple-like objects.
         *
         * This function accepts either two or three arguments. If three arguments are given, the second is the initial
         * state and the third a tuple-like object to fold. If only two arguments are given, the second is a tuple-like
         * object where the first element acts as the initial state.
         *
         * @tparam Fun Binary function type.
         * @tparam Arg Either the initial state if three arguments are given or the tuple to fold if two arguments are
         * given.
         * @tparam Args The tuple type to fold (if three arguments are given).
         *
         * @param fun Binary function object.
         * @param arg Either the initial state if three arguments are given or the tuple to fold if two arguments are
         * given.
         * @param args The tuple to fold (if three arguments are given).
         *
         * Example:
         * @code
         * auto tup = std::make_tuple(1, 2, 3);
         *
         * // Three arguments
         * auto res = fold(std::plus<int>{}, 0, tup);
         * // res == 6
         *
         * // Two arguments
         * auto res2 = fold(std::plus<int>{}, tup);
         * // res2 == 6
         * @endcode
         */
        template < class Fun, class Arg, class... Args >
        auto fold(Fun &&fun, Arg &&arg, Args &&... args) GT_AUTO_RETURN(
            _impl::fold_f< Fun >{std::forward< Fun >(fun)}(std::forward< Arg >(arg), std::forward< Args >(args)...));

        /**
         * @brief Returns a functor that performs a left fold on tuple-like objects.
         *
         * The returned functor accepts either one or two arguments. If two arguments are given, the first is the
         * initial state and the second a tuple-like object to fold. If only one argument is given, the argument must be
         * a tuple-like object where the first element acts as the initial state.
         *
         * @tparam Fun Binary function type.
         * @param fun Binary function object.
         *
         * Example:
         * @code
         * auto tup = std::make_tuple(1, 2, 3);
         * auto folder = fold(std::plus<int>{});
         *
         * // Three arguments
         * auto res = folder(0, tup);
         * // res == 6
         *
         * // Two arguments
         * auto res2 = folder(tup);
         * // res2 == 6
         * @endcode
         */
        template < class Fun >
        constexpr _impl::fold_f< Fun > fold(Fun fun) {
            return {std::move(fun)};
        }

        /**
         * @brief Returns a functor that replaces reference types by value types in a tuple
         *
         * Example:
         * @code
         * auto copyfun = deep_copy();
         * int foo = 3;
         * std::tuple<int&> tup(foo);
         * auto tupcopy = copyfun(tup);
         * ++foo;
         * // tup == {4}, tupcopy == {3}
         * @endcode
         */
        inline constexpr _impl::transform_f< clone > deep_copy() { return {}; }

        /**
         * @brief Replaces reference types by value types in a tuple.
         *
         * @tparam Tup Tuple-like type.
         * @param tup Tuple-like object, possibly containing references.
         *
         * Example:
         * @code
         * int foo = 3;
         * std::tuple<int&> tup(foo);
         * auto tupcopy = deep_copy(tup);
         * ++foo;
         * // tup == {4}, tupcopy == {3}
         * @endcode
         */
        template < class Tup >
        auto deep_copy(Tup &&tup) GT_AUTO_RETURN(deep_copy()(std::forward< Tup >(tup)));
    }
    /** @} */
    /** @} */
}
