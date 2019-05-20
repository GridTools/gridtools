/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \addtogroup common
    @{
*/
/** \addtogroup tupleutils Utilities for Tuples
    @{
*/

/**
 *  @file
 *
 *  Here is a set of algorithms that are defined on "tuple like" structures
 *
 *  The formal definition of the "tuple like" concept:
 *  A type `%T` satisfies the concept if:
 *    - it is move constructible;
 *
 *    - it can be constructed element wise using brace initializer syntax;
 *      [example: `my_triple<T1, T2, T3> val = {elem0, elem1, elem2};` ]
 *
 *    - a function `tuple_to_types(T)` can be found from the `::gridtools::tuple_util::traits` namespace [via ADL or
 *      directly in this namespace]. This function should return a type, which is instantiation of a template
 *      parameterized on types. The actual parameters of that instantiation is interpreted as a types of elements of the
 *      "tuple like". Simply speaking `tuple_to_types` returns a type list of the types of `T` elements.
 *      Note that this function (and for others in this concept definition as well) will be never called. It is enough
 *      to just declare it. Example:
 *      \code
 *      // for the simple "tuple_like"'s it is enough to return itself from tuple_to_types
 *      template <class T, class U, class Q>
 *      my_triple<T, U, Q> tuple_to_types(my_triple<T, U, Q>);
 *      \endcode
 *
 *    - a function `tuple_from_types(T)` can be found from the `::gridtools::tuple_util::traits` namespace [via ADL or
 *      directly in this namespace]. This function should return a type which is a meta class [in the terms of `meta`
 *      library]. This meta class should contain a meta function that takes types elements that we are going to pass to
 *      brace initializer and returns a type [satisfying the same concept] that can accept such a list. Example:
 *      \code
 *      struct my_triple_from_types {
 *          template <class T, class U, class Q>
 *          using apply = my_triple<T, U, Q>;
 *      };
 *      template <class T, class U, class Q>
 *      my_triple_from_types tuple_from_types(my_triple<T, U, Q>);
 *      \endcode
 *
 *    - a function `tuple_getter(T)` can be found from the `::gridtools::tuple_util::traits` namespace [via ADL or
 *      directly in this namespace]. This function should return a type that has a static template (on size_t) member
 *      function called `get<N>` that accepts `T` by some reference and returns the Nth element of `T`. Example:
 *      \code
 *      struct my_triple_getter {
 *          template <size_t N, class T, class U, class Q, enable_if_t<N == 0, int> = 0>
 *          static T get(my_triple<T, U, Q> obj) { return obj.first; }
 *          ...
 *      };
 *      template <class T, class U, class Q>
 *      my_triple_getter tuple_getter(my_triple<T, U, Q>);
 *      \endcode
 *
 *  If the opposite is not mentioned explicitly, the algorithms produce tuples of references. L-value or R-value
 *  depending on algorithm input.
 *
 *  Almost all algorithms are defined in two forms:
 *    1) conventional template functions;
 *    2) functions that return generic functors
 *
 *  For example you can do:
 *  \code
 *    auto ints = transform([](int x) {return x;}, input);
 *  \endcode
 *  or you can:
 *  \code
 *    auto convert_to_ints = transform([](int x) {return x;});
 *    auto ints = convert_to_ints(input);
 *  \endcode
 *
 *  The second form is more composable. For example if the input is a tuple of tuples of whatever and you need a
 *  tuple of tuple of tuple of integers you can do it in one expression:
 *  \code
 *  auto out = transform(transform([](int x) {return x;}), input);
 *  \endcode
 *
 *
 *  TODO list
 *  =========
 *  - add for_each_index
 *  - add filter
 *
 */

#ifndef GT_TARGET_ITERATING
//// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_TUPLE_UTIL_HPP_
#define GT_COMMON_TUPLE_UTIL_HPP_

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../meta.hpp"
#include "defs.hpp"
#include "functional.hpp"
#include "generic_metafunctions/implicit_cast.hpp"
#include "generic_metafunctions/utility.hpp"
#include "host_device.hpp"

#define GT_TUPLE_UTIL_FORWARD_CTORS_TO_MEMBER(class_name, member_name)                                              \
    template <class... Args, enable_if_t<std::is_constructible<decltype(member_name), Args &&...>::value, int> = 0> \
    GT_CONSTEXPR GT_FUNCTION class_name(Args &&... args) noexcept : member_name{wstd::forward<Args>(args)...} {}    \
    GT_DECLARE_DEFAULT_EMPTY_CTOR(class_name);                                                                      \
    class_name(class_name const &) = default;                                                                       \
    class_name(class_name &&) = default;                                                                            \
    class_name &operator=(class_name const &) = default;                                                            \
    class_name &operator=(class_name &&) = default

#define GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(class_name, member_name)                                                \
    struct class_name##_tuple_util_getter {                                                                            \
        template <size_t I>                                                                                            \
        static GT_CONSTEXPR GT_FUNCTION auto get(class_name const &obj)                                                \
            GT_AUTO_RETURN(tuple_util::host_device::get<I>(obj.member_name));                                          \
        template <size_t I>                                                                                            \
        static GT_FUNCTION auto get(class_name &obj) GT_AUTO_RETURN(tuple_util::host_device::get<I>(obj.member_name)); \
        template <size_t I>                                                                                            \
        static GT_CONSTEXPR GT_FUNCTION auto get(class_name &&obj)                                                     \
            GT_AUTO_RETURN(tuple_util::host_device::get<I>(wstd::move(obj).member_name));                              \
    };                                                                                                                 \
    friend class_name##_tuple_util_getter tuple_getter(class_name const &) { return {}; }                              \
    static_assert(1, "")

namespace gridtools {
    namespace tuple_util {

        /// @cond
        namespace traits {
            namespace _impl {

                template <class... Ts>
                struct deduce_array_type : std::common_type<Ts...> {};

                template <>
                struct deduce_array_type<> {
                    using type = meta::lazy::id<void>;
                };

                /// implementation of the `from_types`  for `std::array`
                //
                struct array_from_types {
                    template <class... Ts>
                    using apply = meta::id<std::array<typename deduce_array_type<Ts...>::type, sizeof...(Ts)>>;
                };

                /// getter for the standard "tuple like" entities: `std::tuple`, `std::pair` and `std::array`
                //
                struct std_getter {
                    template <size_t I, class T>
                    GT_FORCE_INLINE static constexpr auto get(T &&obj) noexcept GT_AUTO_RETURN(
                        std::get<I>(wstd::forward<T>(obj)));
                };
            } // namespace _impl

            // start of builtin adaptations

            // to_types

            // generic `tuple_to_types` that works for `std::tuple`, `std::pair` and its clones.
            // It just returns an argument;
            template <template <class...> class L, class... Ts>
            L<Ts...> tuple_to_types(L<Ts...>);

            // `std::array` specialization. Returns the type from array repeated N times.
            template <class T, size_t N>
            meta::repeat_c<N, T> tuple_to_types(std::array<T, N>);

            // from_types

            // generic `tuple_from_types` that works for `std::tuple`, `std::pair` and its clones.
            // meta constructor 'L' is extracted and used to build the new "tuple like"
            template <template <class...> class L, class... Ts>
            meta::ctor<L<Ts...>> tuple_from_types(L<Ts...>);

            // arrays specialization.
            template <class T, size_t N>
            _impl::array_from_types tuple_from_types(std::array<T, N>);

            // getter

            // all `std` "tuple_like"s use `std::get`
            template <class... Ts>
            _impl::std_getter tuple_getter(std::tuple<Ts...>);
            template <class T, class U>
            _impl::std_getter tuple_getter(std::pair<T, U>);
            template <class T, size_t N>
            _impl::std_getter tuple_getter(std::array<T, N>);

            // end of builtin adaptations

            // Here ADL definitions of `tuple_*` functions are picked up
            // The versions in this namespace will be chosen if nothing is found by `ADL`.
            // it is important to have all builtins above this line.

            template <class T>
            decltype(tuple_getter(std::declval<T>())) get_getter(T);
            template <class T>
            decltype(tuple_to_types(std::declval<T>())) get_to_types(T);
            template <class T>
            decltype(tuple_from_types(std::declval<T>())) get_from_types(T);

#if GT_BROKEN_TEMPLATE_ALIASES
#define GT_TUPLE_UTIL_DEFINE_SAFE_ALIAS(name)                                                          \
    template <class, class = void>                                                                     \
    struct name;                                                                                       \
    template <class T>                                                                                 \
    struct name<T, void_t<decltype(::gridtools::tuple_util::traits::get_##name(std::declval<T>()))>> { \
        using type = decltype(::gridtools::tuple_util::traits::get_##name(std::declval<T>()));         \
    }
#else
#define GT_TUPLE_UTIL_DEFINE_SAFE_ALIAS(name) \
    template <class T>                        \
    using name = decltype(::gridtools::tuple_util::traits::get_##name(std::declval<T>()))
#endif
            GT_TUPLE_UTIL_DEFINE_SAFE_ALIAS(getter);
            GT_TUPLE_UTIL_DEFINE_SAFE_ALIAS(to_types);
            GT_TUPLE_UTIL_DEFINE_SAFE_ALIAS(from_types);
#undef GT_TUPLE_UTIL_DEFINE_SAFE_ALIAS
        } // namespace traits
        /// @endcond

        ///  Generalization of std::tuple_size
        //
        template <class T>
        using size = meta::length<traits::to_types<T>>;

        ///  Generalization of std::tuple_element
        //
        GT_META_LAZY_NAMESPACE {
            template <size_t I, class T>
            using element = meta::lazy::at_c<traits::to_types<T>, I>;
        }
        GT_META_DELEGATE_TO_LAZY(element, (size_t I, class T), (I, T));

        // Here goes the stuff that is common for all targets (meta functions)
        namespace _impl {

            template <class T>
            using to_types = traits::to_types<decay_t<T>>;

            template <class Sample, class Types, class FromTypesMetaClass = traits::from_types<decay_t<Sample>>>
            using from_types = meta::rename<FromTypesMetaClass::template apply, Types>;

            enum class ref_kind { rvalue, lvalue, const_lvalue };

            template <class>
            struct get_ref_kind;

            template <class T>
            struct get_ref_kind<T &&> : std::integral_constant<ref_kind, ref_kind::rvalue> {};

            template <class T>
            struct get_ref_kind<T &> : std::integral_constant<ref_kind, ref_kind::lvalue> {};

            template <class T>
            struct get_ref_kind<T const &> : std::integral_constant<ref_kind, ref_kind::const_lvalue> {};

            GT_META_LAZY_NAMESPACE {
                template <ref_kind Kind, class Dst>
                struct add_ref;

                template <class T>
                struct add_ref<ref_kind::rvalue, T> : std::add_rvalue_reference<T> {};

                template <class T>
                struct add_ref<ref_kind::lvalue, T> : std::add_lvalue_reference<T> {};

                template <class T>
                struct add_ref<ref_kind::const_lvalue, T> : std::add_lvalue_reference<add_const_t<T>> {};
            }
            GT_META_DELEGATE_TO_LAZY(add_ref, (ref_kind Kind, class Dst), (Kind, Dst));

            template <ref_kind Kind>
            struct get_accessor {
                template <class T>
                using apply = add_ref<Kind, T>;
            };

            template <class Fun>
            struct get_fun_result {
                template <class... Ts>
                using apply = meta::id<decltype(std::declval<Fun>()(std::declval<Ts>()...))>;
            };

            template <class Tup>
            using get_accessors =
                meta::transform<get_accessor<get_ref_kind<Tup>::value>::template apply, to_types<Tup>>;

            template <class D, class... Ts>
            struct make_array_helper {
                using type = D;
            };

            template <class... Ts>
            struct make_array_helper<void, Ts...> : std::common_type<Ts...> {};

            template <template <class...> class L>
            struct to_tuple_converter_helper {
                template <class... Ts>
                using apply = meta::id<L<Ts...>>;
            };

            template <template <class, size_t> class Arr, class D>
            struct to_array_converter_helper {
                template <class... Ts>
                using apply = meta::id<Arr<typename make_array_helper<D, Ts...>::type, sizeof...(Ts)>>;
            };

        } // namespace _impl
    }     // namespace tuple_util
} // namespace gridtools

// Now it's time to generate host/device/host_device stuff
#define GT_FILENAME <gridtools/common/tuple_util.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif // GT_COMMON_TUPLE_UTIL_HPP_
#else  // GT_TARGET_ITERATING

namespace gridtools {
    namespace tuple_util {
        GT_TARGET_NAMESPACE {
            /**
             * @brief Tuple element accessor like std::get.
             *
             * @tparam I Element index.
             * @tparam T Tuple-like type.
             * @param obj Tuple-like object.
             */
            template <size_t I, class T, class Getter = traits::getter<decay_t<T>>>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto get(T && obj) noexcept GT_AUTO_RETURN(
                Getter::template get<I>(wstd::forward<T>(obj)));

            template <size_t I>
            struct get_nth_f {
                template <class T, class Getter = traits::getter<decay_t<T>>>
                GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto operator()(T &&obj) const
                    noexcept GT_AUTO_RETURN(Getter::template get<I>(wstd::forward<T>(obj)));
            };

            // Let as use `detail` for internal namespace of the target dependent namespace.
            // This way we can refer `_impl::foo` for the entities that are independent on the target and
            // `detail::bar` for the target dependent ones.
            namespace detail {
                using _impl::from_types;
                using _impl::get_accessors;
                using _impl::get_fun_result;
                using _impl::to_types;

                template <size_t I>
                struct transform_elem_f {
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 9
                    // CAUTION!! CUDA8 barely understands this. If you just GT_AUTO_RETURN it goes nuts with mysterious
                    // error message; if you replace the inner result_of_t to typename std::result_of<...>::type it
                    // fails as well. Alternatively you can also write: auto operator()(Fun &&fun, Tups &&... tups)
                    // const
                    // -> typename std::result_of<Fun&&(decltype(get< I >(wstd::forward< Tups >(tups)))...)>::type
                    template <class Fun, class... Tups>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR
                        typename std::result_of<Fun && (result_of_t<get_nth_f<I>(Tups &&)>...)>::type
                        operator()(Fun &&fun, Tups &&... tups) const {
                        return wstd::forward<Fun>(fun)(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tups>(tups))...);
                    }
#elif (defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1800) || \
    (defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ <= 10)
                    template <class Fun, class Tup>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto operator()(Fun &&fun, Tup &&tup) const GT_AUTO_RETURN(
                        wstd::forward<Fun>(fun)(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup>(tup))));
                    template <class Fun, class Tup1, class Tup2>
                    GT_TARGET auto operator()(Fun &&fun, Tup1 &&tup1, Tup2 &&tup2) const GT_AUTO_RETURN(
                        wstd::forward<Fun>(fun)(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup1>(tup1)),
                            GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup2>(tup2))));
                    template <class Fun, class Tup1, class Tup2, class Tup3>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto operator()(
                        Fun &&fun, Tup1 &&tup1, Tup2 &&tup2, Tup3 &&tup3) const
                        GT_AUTO_RETURN(
                            wstd::forward<Fun>(fun)(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup1>(tup1)),
                                GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup2>(tup2)),
                                GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup3>(tup3))));
#else
                    template <class Fun, class... Tups>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto operator()(Fun &&fun, Tups &&... tups) const
                        GT_AUTO_RETURN(
                            wstd::forward<Fun>(fun)(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tups>(tups))...));
#endif
                };

                template <class I>
                using get_transform_generator = meta::id<transform_elem_f<I::value>>;

                template <class GeneratorList, class Res>
                struct generate_f;
                template <template <class...> class L, class... Generators, class Res>
                struct generate_f<L<Generators...>, Res> {
                    template <class... Args>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Args &&... args) const {
                        return Res{Generators{}(wstd::forward<Args>(args)...)...};
                    }
                };

                template <class Fun>
                struct transform_f {
                    template <class... Args>
                    using get_results_t = meta::transform<get_fun_result<Fun>::template apply, Args...>;

                    Fun m_fun;

                    template <class Tup,
                        class... Tups,
                        class Res = from_types<Tup, get_results_t<get_accessors<Tup &&>, get_accessors<Tups &&>...>>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup, Tups &&... tups) const {
                        using generators =
                            meta::transform<get_transform_generator, meta::make_indices_c<size<decay_t<Tup>>::value>>;
                        return generate_f<generators, Res>{}(
                            m_fun, wstd::forward<Tup>(tup), wstd::forward<Tups>(tups)...);
                    }
                };

                template <class Fun>
                struct for_each_adaptor_f {
                    Fun m_fun;
                    template <class... Args>
                    GT_TARGET GT_FORCE_INLINE meta::lazy::id<void> operator()(Args &&... args) const {
                        m_fun(wstd::forward<Args>(args)...);
                        return {};
                    }
                };

                template <class Indices>
                struct apply_to_elements_f;

                template <template <class...> class L, class... Is>
                struct apply_to_elements_f<L<Is...>> {
                    template <class Fun, class... Tups>
                    GT_TARGET GT_FORCE_INLINE auto operator()(Fun &&fun, Tups &&... tups) const
                        GT_AUTO_RETURN(wstd::forward<Fun>(fun)(
                            GT_TARGET_NAMESPACE_NAME::get<Is::value>(wstd::forward<Tups>(tups))...));
                };

                template <class>
                struct for_each_in_cartesian_product_impl_f;

                template <template <class...> class Outer, class... Inners>
                struct for_each_in_cartesian_product_impl_f<Outer<Inners...>> {
                    template <class Fun, class... Tups>
                    GT_TARGET GT_FORCE_INLINE void operator()(Fun &&fun, Tups &&... tups) const {
                        void((int[]){
                            (apply_to_elements_f<Inners>{}(wstd::forward<Fun>(fun), wstd::forward<Tups>(tups)...),
                                0)...});
                    }
                };

                template <class Fun>
                struct for_each_in_cartesian_product_f {
                    Fun m_fun;
                    template <class... Tups>
                    GT_TARGET GT_FORCE_INLINE void operator()(Tups &&... tups) const {
                        for_each_in_cartesian_product_impl_f<
                            meta::cartesian_product<meta::make_indices_c<size<decay_t<Tups>>::value>...>>{}(
                            m_fun, wstd::forward<Tups>(tups)...);
                    }
                };

                struct flatten_f {
                    template <size_t OuterI, size_t InnerI>
                    struct generator_f {
                        template <class Tup>
                        GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto operator()(Tup &&tup) const
                            GT_AUTO_RETURN(GT_TARGET_NAMESPACE_NAME::get<InnerI>(
                                GT_TARGET_NAMESPACE_NAME::get<OuterI>(wstd::forward<Tup>(tup))));
                    };

                    template <class OuterI, class InnerI>
                    using get_generator = meta::id<generator_f<OuterI::value, InnerI::value>>;

                    template <class OuterI, class InnerTup>
                    using get_inner_generators =
                        meta::transform<meta::bind<get_generator, OuterI, meta::_1>::template apply,
                            meta::make_indices_for<InnerTup>>;

                    template <class Tup,
                        class Accessors = meta::transform<get_accessors, get_accessors<Tup &&>>,
                        class First = meta::first<to_types<Tup>>,
                        class Res = from_types<First, meta::flatten<Accessors>>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup) const {
                        GT_STATIC_ASSERT(size<decay_t<Tup>>::value != 0, "can not flatten empty tuple");
                        using generators = meta::flatten<
                            meta::transform<get_inner_generators, meta::make_indices_for<Accessors>, Accessors>>;
                        return generate_f<generators, Res>{}(wstd::forward<Tup>(tup));
                    }
                };

                template <size_t N>
                struct drop_front_f {
                    template <class I>
                    using get_drop_front_generator = meta::id<get_nth_f<N + I::value>>;

                    template <class Tup,
                        class Accessors = get_accessors<Tup &&>,
                        class Res = from_types<Tup, meta::drop_front_c<N, Accessors>>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup) const {
                        using generators =
                            meta::transform<get_drop_front_generator, meta::make_indices_c<size<Accessors>::value - N>>;
                        return generate_f<generators, Res>{}(wstd::forward<Tup>(tup));
                    }
                };

                template <class, class>
                struct push_back_impl_f;

                template <template <class T, T...> class L, class Int, Int... Is, class Res>
                struct push_back_impl_f<L<Int, Is...>, Res> {
                    template <class Tup, class... Args>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup, Args &&... args) const {
                        return Res{GT_TARGET_NAMESPACE_NAME::get<Is>(wstd::forward<Tup>(tup))...,
                            wstd::forward<Args>(args)...};
                    }
                };

                struct push_back_f {
                    template <class Tup,
                        class... Args,
                        class Accessors = get_accessors<Tup &&>,
                        class Res = from_types<Tup, meta::push_back<Accessors, Args &&...>>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup, Args &&... args) const {
                        return push_back_impl_f<meta::make_index_sequence<size<Accessors>::value>, Res>{}(
                            wstd::forward<Tup>(tup), wstd::forward<Args>(args)...);
                    }
                };

                template <class, class>
                struct push_front_impl_f;

                template <template <class T, T...> class L, class Int, Int... Is, class Res>
                struct push_front_impl_f<L<Int, Is...>, Res> {
                    template <class Tup, class... Args>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup, Args &&... args) const {
                        return Res{wstd::forward<Args>(args)...,
                            GT_TARGET_NAMESPACE_NAME::get<Is>(wstd::forward<Tup>(tup))...};
                    }
                };

                struct push_front_f {
                    template <class Tup,
                        class... Args,
                        class Accessors = get_accessors<Tup &&>,
                        class Res = from_types<Tup, meta::push_front<Accessors, Args &&...>>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup, Args &&... args) const {
                        return push_front_impl_f<meta::make_index_sequence<size<Accessors>::value>, Res>{}(
                            wstd::forward<Tup>(tup), wstd::forward<Args>(args)...);
                    }
                };

                template <class Fun>
                struct fold_f {
#if GT_BROKEN_TEMPLATE_ALIASES
                    template <class S, class T>
                    struct meta_fun : get_fun_result<Fun>::template apply<S, T> {};
#else
                    template <class S, class T>
                    using meta_fun = typename get_fun_result<Fun>::template apply<S, T>;
#endif
                    Fun m_fun;

                    template <size_t I, size_t N, class State, class Tup, enable_if_t<I == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR State impl(State &&state, Tup &&) const {
                        return state;
                    }

                    template <size_t I, size_t N, class State, class Tup, enable_if_t<I + 1 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto impl(State &&state, Tup &&tup) const GT_AUTO_RETURN(
                        m_fun(wstd::forward<State>(state), GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup>(tup))));

                    template <size_t I, size_t N, class State, class Tup, enable_if_t<I + 2 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto impl(State &&state, Tup &&tup) const
                        GT_AUTO_RETURN(m_fun(m_fun(wstd::forward<State>(state),
                                                 GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup>(tup))),
                            GT_TARGET_NAMESPACE_NAME::get<I + 1>(wstd::forward<Tup>(tup))));

                    template <size_t I, size_t N, class State, class Tup, enable_if_t<I + 3 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto impl(State &&state, Tup &&tup) const
                        GT_AUTO_RETURN(m_fun(m_fun(m_fun(wstd::forward<State>(state),
                                                       GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup>(tup))),
                                                 GT_TARGET_NAMESPACE_NAME::get<I + 1>(wstd::forward<Tup>(tup))),
                            GT_TARGET_NAMESPACE_NAME::get<I + 2>(wstd::forward<Tup>(tup))));

                    template <size_t I, size_t N, class State, class Tup, enable_if_t<I + 4 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto impl(State &&state, Tup &&tup) const
                        GT_AUTO_RETURN(m_fun(m_fun(m_fun(m_fun(wstd::forward<State>(state),
                                                             GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup>(tup))),
                                                       GT_TARGET_NAMESPACE_NAME::get<I + 1>(wstd::forward<Tup>(tup))),
                                                 GT_TARGET_NAMESPACE_NAME::get<I + 2>(wstd::forward<Tup>(tup))),
                            GT_TARGET_NAMESPACE_NAME::get<I + 3>(wstd::forward<Tup>(tup))));

                    template <size_t I,
                        size_t N,
                        class State,
                        class Tup,
                        class AllAccessors = get_accessors<Tup &&>,
                        class Accessors = meta::drop_front_c<I, AllAccessors>,
                        class Res = meta::lfold<meta_fun, State &&, Accessors>,
                        enable_if_t<(I + 4 < N), int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res impl(State &&state, Tup &&tup) const {
                        return impl<I + 5, N>(
                            m_fun(m_fun(m_fun(m_fun(m_fun(wstd::forward<State>(state),
                                                        GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup>(tup))),
                                                  GT_TARGET_NAMESPACE_NAME::get<I + 1>(wstd::forward<Tup>(tup))),
                                            GT_TARGET_NAMESPACE_NAME::get<I + 2>(wstd::forward<Tup>(tup))),
                                      GT_TARGET_NAMESPACE_NAME::get<I + 3>(wstd::forward<Tup>(tup))),
                                GT_TARGET_NAMESPACE_NAME::get<I + 4>(wstd::forward<Tup>(tup))),
                            wstd::forward<Tup>(tup));
                    }

                    template <class State,
                        class Tup,
                        class Accessors = get_accessors<Tup &&>,
                        class Res = meta::lfold<meta_fun, State &&, Accessors>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(State &&state, Tup &&tup) const {
                        return impl<0, size<decay_t<Tup>>::value>(wstd::forward<State>(state), wstd::forward<Tup>(tup));
                    }

                    template <class Tup,
                        class AllAccessors = get_accessors<Tup &&>,
                        class StateAccessor = meta::first<AllAccessors>,
                        class Accessors = meta::drop_front_c<1, AllAccessors>,
                        class Res = meta::lfold<meta_fun, StateAccessor, Accessors>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup) const {
                        return impl<1, size<decay_t<Tup>>::value>(
                            GT_TARGET_NAMESPACE_NAME::get<0>(wstd::forward<Tup>(tup)), wstd::forward<Tup>(tup));
                    }
                };

                template <class Fun>
                struct all_of_f {
                    Fun m_fun;

                    template <size_t I, size_t N, class... Tups, enable_if_t<I == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR bool impl(Tups &&...) const {
                        return true;
                    }

                    template <size_t I, size_t N, class... Tups, enable_if_t<I + 1 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR bool impl(Tups &&... tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tups>(tups))...);
                    }

                    template <size_t I, size_t N, class... Tups, enable_if_t<I + 2 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR bool impl(Tups &&... tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 1>(wstd::forward<Tups>(tups))...);
                    }

                    template <size_t I, size_t N, class... Tups, enable_if_t<I + 3 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR bool impl(Tups &&... tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 1>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 2>(wstd::forward<Tups>(tups))...);
                    }

                    template <size_t I, size_t N, class... Tups, enable_if_t<I + 4 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR bool impl(Tups &&... tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 1>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 2>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 3>(wstd::forward<Tups>(tups))...);
                    }

                    template <size_t I, size_t N, class... Tups, enable_if_t<(I + 4 < N), int> = 0>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR bool impl(Tups &&... tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 1>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 2>(wstd::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 3>(wstd::forward<Tups>(tups))...) &&
                               impl<I + 4, N>(wstd::forward<Tups>(tups)...);
                    }

                    template <class Tup, class... Tups>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR bool operator()(Tup &&tup, Tups &&... tups) const {
                        return impl<0, size<decay_t<Tup>>::value>(
                            wstd::forward<Tup>(tup), wstd::forward<Tups>(tups)...);
                    }
                };

                template <class To, class Index>
                struct implicit_convert_to_f {
                    using type = implicit_convert_to_f;
                    template <class Tup>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR To operator()(Tup &&tup) const {
                        return GT_TARGET_NAMESPACE_NAME::get<Index::value>(tup);
                    }
                };

                template <class DstFromTypesMetaClass>
                struct convert_to_f {
                    template <class Tup,
                        class ToTypes = to_types<Tup>,
                        class Res = meta::rename<DstFromTypesMetaClass::template apply, ToTypes>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup const &tup) const {
                        using generators_t =
                            meta::transform<implicit_convert_to_f, to_types<Res>, meta::make_indices_for<ToTypes>>;
                        return generate_f<generators_t, Res>{}(tup);
                    }
                };

                struct transpose_f {
                    template <class Tup>
                    struct get_inner_tuple_f {
                        template <class Types>
                        using apply = from_types<Tup, Types>;
                    };

                    template <class I>
                    using get_generator = meta::id<transform_f<get_nth_f<I::value>>>;

                    template <class Tup,
                        class First = meta::first<to_types<Tup>>,
                        class Accessors = meta::transform<get_accessors, get_accessors<Tup &&>>,
                        class Types = meta::transpose<Accessors>,
                        class InnerTuples = meta::transform<get_inner_tuple_f<Tup>::template apply, Types>,
                        class Res = from_types<First, InnerTuples>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup) const {
                        GT_STATIC_ASSERT(
                            tuple_util::size<decay_t<Tup>>::value, "tuple_util::transpose input should not be empty");
                        using inner_indices_t = meta::make_indices_for<to_types<First>>;
                        using generators_t = meta::transform<get_generator, inner_indices_t>;
                        return generate_f<generators_t, Res>{}(wstd::forward<Tup>(tup));
                    }
                };

                struct reverse_f {
                    template <class N>
                    struct generator_f {
                        template <class I>
                        using apply = meta::id<get_nth_f<N::value - 1 - I::value>>;
                    };

                    template <class Tup,
                        class Accessors = get_accessors<Tup &&>,
                        class Res = from_types<Tup, meta::reverse<Accessors>>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup) const {
                        using n_t = size<decay_t<Tup>>;
                        using generators_t = meta::transform<generator_f<n_t>::template apply, meta::make_indices<n_t>>;
                        return generate_f<generators_t, Res>{}(wstd::forward<Tup>(tup));
                    }
                };

                template <size_t I>
                struct insert_tup_generator_f {
                    using type = insert_tup_generator_f;

                    template <class Tup, class Val>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto operator()(Tup &&tup, Val &&) const
                        GT_AUTO_RETURN(GT_TARGET_NAMESPACE_NAME::get<I>(wstd::forward<Tup>(tup)));
                };

                struct insert_val_generator_f {
                    using type = insert_val_generator_f;
                    template <class Tup, class Val>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Val operator()(Tup &&, Val &&val) const {
                        return wstd::forward<Val>(val);
                    }
                };

                template <size_t N, class Val>
                struct insert_f {
                    Val m_val;

                    template <class I>
                    using get_generator =
                        meta::if_c <
                        I::value<N,
                            insert_tup_generator_f<I::value>,
                            meta::if_c<I::value == N, insert_val_generator_f, insert_tup_generator_f<I::value - 1>>>;

                    template <class Tup,
                        class Accessors = get_accessors<Tup &&>,
                        class Types = meta::insert_c<N, Accessors, Val>,
                        class Res = from_types<Tup, Types>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res operator()(Tup &&tup) const {
                        using generators_t = meta::transform<get_generator, meta::make_indices_for<Types>>;
                        return generate_f<generators_t, Res>{}(wstd::forward<Tup>(tup), m_val);
                    }
                };
            } // namespace detail

            /**
             * @brief Transforms each tuple element by a function.
             *
             * Transformations with functions with more than one argument are supported by passing multiple tuples of
             * the same size.
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
            template <class Fun, class Tup, class... Tups>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto transform(Fun && fun, Tup && tup, Tups && ... tups)
                GT_AUTO_RETURN(detail::transform_f<Fun>{wstd::forward<Fun>(fun)}(
                    wstd::forward<Tup>(tup), wstd::forward<Tups>(tups)...));

            /**
             * @brief Returns a functor that transforms each tuple element by a function.
             *
             * Composable version of `transform` that returns a functor which can be invoked with (one or multiple)
             * tuples.
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
            template <class Fun>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::transform_f<Fun> transform(Fun fun) {
                return {wstd::move(fun)};
            }

            /**
             * @brief Calls a function for each element in a tuple.
             *
             * Functions with more than one argument are supported by passing multiple tuples of the same size. If only
             * a function but no tuples are passed, a composable functor is returned.
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
            template <class Fun, class Tup, class... Tups>
            GT_TARGET GT_FORCE_INLINE void for_each(Fun && fun, Tup && tup, Tups && ... tups) {
                transform(detail::for_each_adaptor_f<Fun>{wstd::forward<Fun>(fun)},
                    wstd::forward<Tup>(tup),
                    wstd::forward<Tups>(tups)...);
            }

            /**
             * @brief Returns a functor that calls a function for each element in a tuple.
             *
             * Composable version of `for_each` that returns a functor which can be invoked with (one or multiple)
             * tuples.
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
            template <class Fun>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::transform_f<detail::for_each_adaptor_f<Fun>> for_each(
                Fun fun) {
                return {{wstd::move(fun)}};
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
            template <class Fun, class Tup, class... Tups>
            GT_TARGET GT_FORCE_INLINE void for_each_in_cartesian_product(Fun && fun, Tup && tup, Tups && ... tups) {
                detail::for_each_in_cartesian_product_f<Fun>{wstd::forward<Fun>(fun)}(
                    wstd::forward<Tup>(tup), wstd::forward<Tups>(tups)...);
            }

            template <class Fun>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::for_each_in_cartesian_product_f<Fun>
            for_each_in_cartesian_product(Fun fun) {
                return {wstd::move(fun)};
            }

            /**
             * @brief Return a functor that flattens a tuple of tuples non-recursively into a single tuple.
             *
             * Flattens only the first two levels of nested tuples into a single level. Does not flatten further levels
             * of nesting.
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
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::flatten_f flatten() { return {}; }

            /**
             * @brief Non-recursively flattens a tuple of tuples into a single tuple.
             *
             * Flattens only the first two levels of nested tuples into a single level. Does not flatten further levels
             * of nesting.
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
            template <class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto flatten(Tup && tup)
                GT_AUTO_RETURN(flatten()(wstd::forward<Tup>(tup)));

            /**
             * @brief Constructs an object from generator functors.
             *
             * `Generators` is a typelist of generator functors. Instances of those types are first default constructed,
             * then invoked with `args` as arguments. The results of those calls are then passed to the constructor of
             * `Res`.
             *
             * @tparam Generators A typelist of functors. All functor types must be default-constructible and callable
             * with arguments of type `Args`.
             * @tparam Res The type that should be constructed.
             * @tparam Args Argument types for the generator functors.
             *
             * @param args Arguments that will be passed to the generator functors.
             *
             * Example:
             * @code
             * // Generators that extract some information from the given arguments (a single std::string in this
             * example) struct ptr_extractor { const char* operator()(std::string const& s) const { return s.data();
             *     }
             * };
             *
             * struct size_extractor {
             *     std::size_t operator()(std::string const& s) const {
             *         return s.size();
             *     }
             * };
             *
             * // We want to generate a pair of a pointer and size, that represents this string in a simple C-style
             * manner std::string s = "Hello World!";
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
            template <class Generators, class Res, class... Args>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Res generate(Args && ... args) {
                return detail::generate_f<Generators, Res>{}(wstd::forward<Args>(args)...);
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
            template <size_t N>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::drop_front_f<N> drop_front() {
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
            template <size_t N, class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto drop_front(Tup && tup)
                GT_AUTO_RETURN(drop_front<N>()(wstd::forward<Tup>(tup)));

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
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::push_back_f push_back() { return {}; }

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
            template <class Tup, class... Args>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto push_back(Tup && tup, Args && ... args)
                GT_AUTO_RETURN(push_back()(wstd::forward<Tup>(tup), wstd::forward<Args>(args)...));

            /**
             * @brief Appends elements to a tuple from the front.
             */
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::push_front_f push_front() { return {}; }

            template <class Tup, class... Args>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto push_front(Tup && tup, Args && ... args)
                GT_AUTO_RETURN(push_front()(wstd::forward<Tup>(tup), wstd::forward<Args>(args)...));

            /**
             * @brief Left fold on tuple-like objects.
             *
             * This function accepts either two or three arguments. If three arguments are given, the second is the
             * initial state and the third a tuple-like object to fold. If only two arguments are given, the second is a
             * tuple-like object where the first element acts as the initial state.
             *
             * @tparam Fun Binary function type.
             * @tparam Arg Either the initial state if three arguments are given or the tuple to fold if two arguments
             * are given.
             * @tparam Args The tuple type to fold (if three arguments are given).
             *
             * @param fun Binary function object.
             * @param arg Either the initial state if three arguments are given or the tuple to fold if two arguments
             * are given.
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
            template <class Fun, class Arg, class... Args>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto fold(Fun && fun, Arg && arg, Args && ... args) GT_AUTO_RETURN(
                detail::fold_f<Fun>{wstd::forward<Fun>(fun)}(wstd::forward<Arg>(arg), wstd::forward<Args>(args)...));

            /**
             * @brief Returns a functor that performs a left fold on tuple-like objects.
             *
             * The returned functor accepts either one or two arguments. If two arguments are given, the first is the
             * initial state and the second a tuple-like object to fold. If only one argument is given, the argument
             * must be a tuple-like object where the first element acts as the initial state.
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
            template <class Fun>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::fold_f<Fun> fold(Fun fun) {
                return {wstd::move(fun)};
            }

            template <class Pred, class... Tups>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto all_of(Pred && pred, Tups && ... tups)
                GT_AUTO_RETURN(detail::all_of_f<Pred>{wstd::forward<Pred>(pred)}(wstd::forward<Tups>(tups)...));

            template <class Pred>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto all_of(Pred && pred)
                GT_AUTO_RETURN(detail::all_of_f<Pred>{wstd::forward<Pred>(pred)});

            /**
             * transposes a `tuple like` of `tuple like`.
             *
             * Ex.
             *   transpose(make<array>(make<array>(1, 2, 3), make<array>(10, 20, 30))) returns the same as
             *   make<array>(make<array>(1, 10), make<array>(2, 20), make<array>(3, 30));
             */
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::transpose_f transpose() { return {}; }

            template <class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto transpose(Tup && tup)
                GT_AUTO_RETURN(transpose()(wstd::forward<Tup>(tup)));

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
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::transform_f<gridtools::GT_TARGET_NAMESPACE_NAME::clone>
            deep_copy() {
                return {};
            }

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
            template <class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto deep_copy(Tup && tup)
                GT_AUTO_RETURN(deep_copy()(wstd::forward<Tup>(tup)));

            namespace detail {
                // in impl as it is not as powerful as std::invoke (does not support invoking member functions)
                template <class Fun, class... Args>
                auto invoke_impl(Fun &&f, Args &&... args)
                    GT_AUTO_RETURN(wstd::forward<Fun>(f)(wstd::forward<Args>(args)...));

                template <class Fun, class Tup, std::size_t... Is>
                GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto apply_impl(Fun &&f, Tup &&tup, meta::index_sequence<Is...>)
                    GT_AUTO_RETURN(invoke_impl(wstd::forward<Fun>(f), get<Is>(wstd::forward<Tup>(tup))...));
            } // namespace detail

            /**
             * @brief Invoke callable f with tuple of arguments.
             *
             * @tparam Fun Functor type.
             * @tparam Tup Tuple-like type.
             * @param tup Tuple-like object containing arguments
             * @param fun Function that should be called with the arguments in tup
             *
             * See std::apply (c++17), with the limitation that it only works for FunctionObjects (not for any Callable)
             */
            template <class Fun, class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto apply(Fun && fun, Tup && tup)
                GT_AUTO_RETURN(detail::apply_impl(wstd::forward<Fun>(fun),
                    wstd::forward<Tup>(tup),
                    meta::make_index_sequence<size<decay_t<Tup>>::value>{}));

            /// Generalization of `std::make_tuple`
            //
            template <template <class...> class L, class... Ts>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR L<Ts...> make(Ts const &... elems) {
                return L<Ts...>{elems...};
            }

            /// Generalization of `std::tie`
            //
            template <template <class...> class L, class... Ts>
            GT_TARGET GT_FORCE_INLINE L<Ts &...> tie(Ts & ... elems) {
                return L<Ts &...>{elems...};
            }

// cuda8 has problems with deducing generic `make`/`tie` in the case of `pair`
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 9
            template <template <class> class L, class T0>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR L<T0> make(T0 const &elem0) {
                return L<T0>{elem0};
            }
            template <template <class, class> class L, class T0, class T1>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR L<T0, T1> make(T0 const &elem0, T1 const &elem1) {
                return L<T0, T1>{elem0, elem1};
            }
            template <template <class, class, class> class L, class T0, class T1, class T2>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR L<T0, T1, T2> make(
                T0 const &elem0, T1 const &elem1, T2 const &elem2) {
                return L<T0, T1, T2>{elem0, elem1, elem2};
            }
            template <template <class, class, class, class> class L, class T0, class T1, class T2, class T3>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR L<T0, T1, T2, T3> make(
                T0 const &elem0, T1 const &elem1, T2 const &elem2, T3 const &elem3) {
                return L<T0, T1, T2, T3>{elem0, elem1, elem2, elem3};
            }

            template <template <class> class L, class T0>
            GT_TARGET GT_FORCE_INLINE L<T0 &> tie(T0 & elem0) {
                return L<T0 &>{elem0};
            }
            template <template <class, class> class L, class T0, class T1>
            GT_TARGET GT_FORCE_INLINE L<T0 &, T1 &> tie(T0 & elem0, T1 & elem1) {
                return L<T0 &, T1 &>{elem0, elem1};
            }
            template <template <class, class, class> class L, class T0, class T1, class T2>
            GT_TARGET GT_FORCE_INLINE L<T0 &, T1 &, T2 &> tie(T0 & elem0, T1 & elem1, T2 & elem2) {
                return L<T0 &, T1 &, T2 &>{elem0, elem1, elem2};
            }
            template <template <class, class, class, class> class L, class T0, class T1, class T2, class T3>
            GT_TARGET GT_FORCE_INLINE L<T0 &, T1 &, T2 &, T3 &> tie(T0 & elem0, T1 & elem1, T2 & elem2, T3 & elem3) {
                return L<T0 &, T1 &, T2 &, T3 &>{elem0, elem1, elem2, elem3};
            }
#endif

            /// Generalization of `std::experimental::make_array`
            //
            template <template <class, size_t> class Arr, class D = void, class... Ts>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Arr<typename _impl::make_array_helper<D, Ts...>::type, sizeof...(Ts)>
            make(Ts && ... elems) {
                using common_type_t = typename _impl::make_array_helper<D, Ts...>::type;
                return {{implicit_cast<common_type_t>(wstd::forward<Ts>(elems))...}};
            }

            /**
             *   The family of `convert_to` functions.
             *
             *   First template parameter could be either some tuple [`std::tuple`, `std::pair` or gridtools clones]
             *   or some array [`std::array` or gridtools clone]
             *   Array vaiants can take additional parameter -- the desired type of array. If it is not provided,
             *   the type is deduced
             *   Runtime parameter is any "tuple like" or none. Variants without runtime parameter return functors as
             *   usual.
             *
             *   Examples of valid appllcations:
             *
             *   convert_to<std::tuple>(some_tuple_like);
             *   convert_to<std::pair>()(some_tuple_like_with_two_elements);
             *   convert_to<std::array>(some_tuple_like);
             *   convert_to<gridtools::array, int>(some_tuple_like);
             */
            template <template <class...> class L>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::convert_to_f<_impl::to_tuple_converter_helper<L>>
            convert_to() {
                return {};
            }

            template <template <class...> class L, class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto convert_to(Tup const &tup)
                GT_AUTO_RETURN(detail::convert_to_f<_impl::to_tuple_converter_helper<L>>{}(tup));

            template <template <class, size_t> class Arr, class D = void>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::convert_to_f<_impl::to_array_converter_helper<Arr, D>>
            convert_to() {
                return {};
            }

            template <template <class, size_t> class Arr, class D = void, class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto convert_to(Tup const &tup)
                GT_AUTO_RETURN((detail::convert_to_f<_impl::to_array_converter_helper<Arr, D>>{}(tup)));

            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::reverse_f reverse() { return {}; }

            template <class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto reverse(Tup && tup)
                GT_AUTO_RETURN(reverse()(wstd::forward<Tup>(tup)));

            template <size_t I, class Val>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR detail::insert_f<I, Val> insert(Val && val) {
                return {wstd::forward<Val>(val)};
            }

            template <size_t I, class Val, class Tup>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto insert(Val && val, Tup && tup)
                GT_AUTO_RETURN(insert<I>(wstd::forward<Val>(val))(wstd::forward<Tup>(tup)));
        }
    } // namespace tuple_util
} // namespace gridtools

#endif // GT_TARGET_ITERATING

/** @} */
/** @} */
