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

#include <boost/preprocessor.hpp>
#include <functional>
#include <type_traits>

#include "../../meta.hpp"

// internal
#define GT_META_INTERNAL_APPLY(fun, args) BOOST_PP_REMOVE_PARENS(fun)<BOOST_PP_REMOVE_PARENS(args)>

// internal
#if GT_BROKEN_TEMPLATE_ALIASES
#define GT_META_INTERNAL_LAZY_PARAM(fun) BOOST_PP_REMOVE_PARENS(fun)
#else
#define GT_META_INTERNAL_LAZY_PARAM(fun) ::gridtools::meta::force<BOOST_PP_REMOVE_PARENS(fun)>::template apply
#endif

namespace gridtools {
    /** \ingroup common
        @{
    */
    /** \ingroup allmeta
        @{
    */
    /** \defgroup meta Meta-Programming Library
        @{
    */

    namespace meta {

        /// placeholder definitions for bind
        template <size_t>
        struct placeholder;

        using _1 = placeholder<0>;
        using _2 = placeholder<1>;
        using _3 = placeholder<2>;
        using _4 = placeholder<3>;
        using _5 = placeholder<4>;
        using _6 = placeholder<5>;
        using _7 = placeholder<6>;
        using _8 = placeholder<7>;
        using _9 = placeholder<8>;
        using _10 = placeholder<9>;

        GT_META_LAZY_NAMESPASE {

            /// some forward declarations is needed here for technical reason.
            template <template <class...> class, class...>
            struct filter;
        }

#if !GT_BROKEN_TEMPLATE_ALIASES

        // 'direct' versions of lazy functions
        template <template <class...> class F, class... Args>
        using filter = typename lazy::filter<F, Args...>::type;

#endif

        GT_META_LAZY_NAMESPASE {

            // internals
            template <template <class...> class Pred>
            struct filter_helper_impl {
                template <class T>
                GT_META_DEFINE_ALIAS(apply, meta::if_, (Pred<T>, list<T>, list<>));
            };

            /**
             *  Filter the list based of predicate
             */
            template <template <class...> class Pred>
            struct filter<Pred> {
                using type = curry_fun<meta::filter, Pred>;
            };
            template <template <class...> class Pred, class List>
            struct filter<Pred, List>
                : flatten<typename concat<list<typename clear<List>::type>,
                      typename transform<filter_helper_impl<Pred>::template apply, List>::type>::type> {};

            // internals
            template <class S, class T>
            GT_META_DEFINE_ALIAS(dedup_step_impl, meta::if_, (st_contains<S, T>, S, typename push_back<S, T>::type));

            /**
             *  Removes duplicates from the List.
             */
            template <class List>
            GT_META_DEFINE_ALIAS(dedup, lfold, (dedup_step_impl, typename clear<List>::type, List));

            template <class List>
            struct first;
            template <template <class...> class L, class T, class... Ts>
            struct first<L<T, Ts...>> {
                using type = T;
            };
            template <class List>
            struct second;
            template <template <class...> class L, class T, class U, class... Ts>
            struct second<L<T, U, Ts...>> {
                using type = U;
            };

            /**
             *   Take Nth element of the List
             */
            template <class List, class N>
            struct at;
            template <class List, template <class I, I> class Const, class Int>
            struct at<List, Const<Int, 0>> : first<List> {};
            template <class List, template <class I, I> class Const, class Int>
            struct at<List, Const<Int, 1>> : second<List> {};
            template <class List, class N>
            struct at
                : second<typename mp_find<typename zip<typename make_indices_for<List>::type, List>::type, N>::type> {};
            template <class List, size_t N>
            GT_META_DEFINE_ALIAS(at_c, at, (List, std::integral_constant<size_t, N>));

            template <class List>
            GT_META_DEFINE_ALIAS(last, at_c, (List, length<List>::value - 1));

            /**
             * return the position of T in the Set. If there is no T, it returns the length of the Set.
             *
             *  @pre All elements in Set are different.
             */
            template <class Set,
                class T,
                class Pair = typename mp_find<typename zip<Set, typename make_indices_for<Set>::type>::type, T>::type>
            struct st_position : if_<std::is_void<Pair>, length<Set>, second<Pair>>::type::type {};

/**
 *  NVCC bug workaround: sizeof... works incorrectly within template alias context.
 */
#ifdef __CUDACC__
            template <class... Ts>
            struct sizeof_3_dots : std::integral_constant<size_t, sizeof...(Ts)> {};

#define GT_SIZEOF_3_DOTS(Ts) ::gridtools::meta::lazy::sizeof_3_dots<Ts...>::value
#else
#define GT_SIZEOF_3_DOTS(Ts) sizeof...(Ts)
#endif

            /**
             *  C++17 drop-offs
             *
             *  Note on conjunction_fast and disjunction_fast are like std counter parts but:
             *    - short-circuiting is not implemented as required by C++17 standard
             *    - amortized complexity is O(1) because of it [in terms of the number of template instantiations].
             */
            template <class... Ts>
            GT_META_DEFINE_ALIAS(conjunction_fast,
                std::is_same,
                (list<std::integral_constant<bool, Ts::value>...>,
                    typename repeat_c<GT_SIZEOF_3_DOTS(Ts), std::true_type>::type));

            template <class... Ts>
            GT_META_DEFINE_ALIAS(disjunction_fast, negation, conjunction_fast<negation<Ts>...>);

            /**
             *   all elements in lists are true
             */
            template <class List>
            struct all : rename<conjunction_fast, List>::type {};

            /**
             *   some element is true
             */
            template <class List>
            struct any : rename<disjunction_fast, List>::type {};

            /**
             *  All elements satisfy predicate
             */
            template <template <class...> class Pred, class List>
            GT_META_DEFINE_ALIAS(all_of, all, (typename transform<Pred, List>::type));

            /**
             *  Some element satisfy predicate
             */
            template <template <class...> class Pred, class List>
            GT_META_DEFINE_ALIAS(any_of, any, (typename transform<Pred, List>::type));

            template <template <class...> class Pred, template <class...> class F>
            struct selective_call_impl {
                template <class Arg>
                GT_META_DEFINE_ALIAS(apply, meta::if_, (Pred<Arg>, GT_META_CALL(F, Arg), Arg));
            };

            template <template <class...> class Pred, template <class...> class F, class List>
            GT_META_DEFINE_ALIAS(selective_transform, transform, (selective_call_impl<Pred, F>::template apply, List));

            /**
             *   True if the template parameter is type list which elements are all different
             */
            template <class>
            struct is_set : std::false_type {};

            template <template <class...> class L, class... Ts>
            struct is_set<L<Ts...>> : std::is_same<L<Ts...>, typename dedup<L<Ts...>>::type> {};

            /**
             *   is_set_fast evaluates to std::true_type if the parameter is a set.
             *   If parameter is not a type list, predicate evaluates to std::false_type.
             *   Compilation fails if the parameter is a type list with duplicated elements.
             *
             *   Its OK to use this predicate in static asserts and not OK in sfinae enablers.
             */
            template <class, class = void>
            struct is_set_fast : std::false_type {};

            template <template <class...> class L, class... Ts>
            struct is_set_fast<L<Ts...>, void_t<decltype(internal::inherit<id<Ts>...>{})>> : std::true_type {};

            /**
             *   replace all Old elements to New within List
             */
            template <class List, class Old, class New>
            GT_META_DEFINE_ALIAS(replace,
                selective_transform,
                (curry<std::is_same, Old>::template apply, meta::always<New>::template apply, List));

            template <class Key>
            struct is_same_key_impl {
                template <class Elem>
                GT_META_DEFINE_ALIAS(apply, std::is_same, (Key, typename first<Elem>::type));
            };

            template <class... NewVals>
            struct replace_values_impl {
                template <class MapElem>
                struct apply;
                template <template <class...> class L, class Key, class... OldVals>
                struct apply<L<Key, OldVals...>> {
                    using type = L<Key, NewVals...>;
                };
            };

            /**
             *  replace element in the map by key
             */
            template <class Map, class Key, class... NewVals>
            GT_META_DEFINE_ALIAS(mp_replace,
                selective_transform,
                (is_same_key_impl<Key>::template apply,
                    GT_META_INTERNAL_LAZY_PARAM(replace_values_impl<NewVals...>::template apply),
                    Map));

            template <class N, class New>
            struct replace_at_impl {
                template <class T, class M>
                struct apply {
                    using type = T;
                };
                template <class T>
                struct apply<T, N> {
                    using type = New;
                };
            };

            /**
             *  replace element at given position
             */
            template <class List, class N, class New>
            GT_META_DEFINE_ALIAS(replace_at,
                transform,
                (GT_META_INTERNAL_LAZY_PARAM((replace_at_impl<N, New>::template apply)),
                    List,
                    typename make_indices_for<List>::type));
            template <class List, size_t N, class New>
            GT_META_DEFINE_ALIAS(replace_at_c, replace_at, (List, std::integral_constant<size_t, N>, New));

            template <class Arg, class... Params>
            struct replace_placeholders_impl : id<Arg> {};

            template <size_t I, class... Params>
            struct replace_placeholders_impl<placeholder<I>, Params...> : at_c<list<Params...>, I> {};

            template <class L>
            struct cartesian_product_step_impl_impl {
                template <class T>
                GT_META_DEFINE_ALIAS(apply,
                    meta::transform,
                    (curry<meta::push_back, T>::template apply, typename rename<list, L>::type));
            };

            template <class S, class L>
            GT_META_DEFINE_ALIAS(cartesian_product_step_impl,
                meta::rename,
                (meta::concat, typename transform<cartesian_product_step_impl_impl<L>::template apply, S>::type));

            template <class... Lists>
            GT_META_DEFINE_ALIAS(cartesian_product, lfold, (cartesian_product_step_impl, list<list<>>, list<Lists...>));

            /**
             *   reverse algorithm.
             *   Complexity is O(N)
             *   Making specializations for the first M allows to divide complexity by M.
             *   At a moment M = 4 (in boost::mp11 implementation it is 10).
             *   For the optimizers: fill free to add more specializations if needed.
             */

            template <class>
            struct reverse;

            template <template <class...> class L>
            struct reverse<L<>> {
                using type = L<>;
            };
            template <template <class...> class L, class T>
            struct reverse<L<T>> {
                using type = L<T>;
            };
            template <template <class...> class L, class T0, class T1>
            struct reverse<L<T0, T1>> {
                using type = L<T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2>
            struct reverse<L<T0, T1, T2>> {
                using type = L<T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3>
            struct reverse<L<T0, T1, T2, T3>> {
                using type = L<T3, T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3, class T4>
            struct reverse<L<T0, T1, T2, T3, T4>> {
                using type = L<T4, T3, T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3, class T4, class... Ts>
            struct reverse<L<T0, T1, T2, T3, T4, Ts...>>
                : push_back<typename reverse<L<Ts...>>::type, T4, T3, T2, T1, T0> {};

            template <class N, class List>
            GT_META_DEFINE_ALIAS(drop_back, reverse, (typename drop_front<N, typename reverse<List>::type>::type));

            template <size_t N, class List>
            GT_META_DEFINE_ALIAS(drop_back_c, reverse, (typename drop_front_c<N, typename reverse<List>::type>::type));

            template <class List>
            GT_META_DEFINE_ALIAS(pop_front, drop_front_c, (1, List));

            template <class List>
            GT_META_DEFINE_ALIAS(pop_back, drop_back_c, (1, List));
        }

        /**
         *  bind for functions
         */
        template <template <class...> class F, class... BoundArgs>
        struct bind {
            template <class... Params>
            GT_META_DEFINE_ALIAS(apply, F, (typename lazy::replace_placeholders_impl<BoundArgs, Params...>::type...));
        };

        using lazy::all;
        using lazy::all_of;
        using lazy::any;
        using lazy::any_of;
        using lazy::conjunction_fast;
        using lazy::disjunction_fast;
        using lazy::is_set;
        using lazy::is_set_fast;
        using lazy::st_position;

#if !GT_BROKEN_TEMPLATE_ALIASES
        // 'direct' versions of lazy functions
        template <class List>
        using first = typename lazy::first<List>::type;
        template <class List>
        using second = typename lazy::second<List>::type;
        template <class List, class N>
        using at = typename lazy::at<List, N>::type;

        template <class List>
        using dedup = typename lazy::dedup<List>::type;
        template <class List, size_t N>
        using at_c = typename lazy::at_c<List, N>::type;
        template <class List>
        using last = typename lazy::last<List>::type;
        template <template <class...> class Pred, template <class...> class F, class List>
        using selective_transform = typename lazy::selective_transform<Pred, F, List>::type;
        template <class List, class Old, class New>
        using replace = typename lazy::replace<List, Old, New>::type;
        template <class Map, class Key, class... NewVals>
        using mp_replace = typename lazy::mp_replace<Map, Key, NewVals...>::type;
        template <class List, class N, class New>
        using replace_at = typename lazy::replace_at<List, N, New>::type;
        template <class List, size_t N, class New>
        using replace_at_c = typename lazy::replace_at_c<List, N, New>::type;
        template <class... Lists>
        using cartesian_product = typename lazy::cartesian_product<Lists...>::type;
        template <class List>
        using reverse = typename lazy::reverse<List>::type;
        template <class List>
        using pop_front = typename lazy::pop_front<List>::type;
        template <class List>
        using pop_back = typename lazy::pop_back<List>::type;
#endif
    } // namespace meta
    /** @} */
    /** @} */
    /** @} */
} // namespace gridtools

#undef GT_META_INTERNAL_LAZY_PARAM
