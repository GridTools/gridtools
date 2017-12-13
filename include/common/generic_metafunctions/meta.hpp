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

/**
 *  @file Minimalistic C++11 metaprogramming library
 *
 *  The metafunctions in this library operate on typelists.
 *  Any instantiation of the template class with class template parameters is treated as a typelist.
 */

#include <type_traits>

#include "gt_integer_sequence.hpp"

namespace gridtools {
    namespace meta {

        template < class... >
        struct list;

        template < class >
        struct is_list : std::false_type {};
        template < template < class... > class L, class... Ts >
        struct is_list< L< Ts... > > : std::true_type {};

        template < class List >
        struct length;
        template < template < class... > class L, class... Ts >
        struct length< L< Ts... > > : std::integral_constant< size_t, sizeof...(Ts) > {};

        /// Extracts "producing template" from the typelist.
        /// I.e ctor<some_instantiation_of_std_tuple>::apply is the alias of std::tuple.
        template < class List >
        struct ctor;
        template < template < class... > class L, class... Ts >
        struct ctor< L< Ts... > > {
            template < class... Us >
            using apply = L< Us... >;
        };

        template < class T >
        struct id {
            using type = T;
        };

        template < class T >
        struct always {
            template < class... >
            using apply = T;
        };

        template < class T, T Val >
        struct always_c {
            template < class U, U >
            using apply = std::integral_constant< T, Val >;
        };

        namespace _impl {
            template < class List >
            struct first;
            template < template < class... > class L, class T, class... Ts >
            struct first< L< T, Ts... > > {
                using type = T;
            };

            template < class List >
            struct second;
            template < template < class... > class L, class T, class U, class... Ts >
            struct second< L< T, U, Ts... > > {
                using type = U;
            };

            template < class List >
            struct third;
            template < template < class... > class L, class T, class U, class Z, class... Ts >
            struct third< L< T, U, Z, Ts... > > {
                using type = Z;
            };

            template < class List, class... Ts >
            struct push_front;
            template < template < class... > class L, class... Us, class... Ts >
            struct push_front< L< Us... >, Ts... > {
                using type = L< Ts..., Us... >;
            };

            template < class List, class... Ts >
            struct push_back;
            template < template < class... > class L, class... Us, class... Ts >
            struct push_back< L< Us... >, Ts... > {
                using type = L< Us..., Ts... >;
            };

            template < template < class... > class F >
            struct rename {
                template < class List >
                struct apply;
                template < template < class... > class From, class... Ts >
                struct apply< From< Ts... > > {
                    using type = F< Ts... >;
                };
                template < class List >
                using apply_t = typename apply< List >::type;
            };

            template < template < class U, U > class F, class ISec >
            struct transform_c;
            template < template < class U, U > class F, class Int, Int... Is >
            struct transform_c< F, gt_integer_sequence< Int, Is... > > {
                using type = gt_integer_sequence< typename F< Int, 0 >::value_type, F< Int, Is >::value... >;
            };

            template < class... Ts >
            struct inherit : Ts... {};

            template < class ISec >
            struct iseq_to_list;
            template < class Int, Int... Is >
            struct iseq_to_list< gt_integer_sequence< Int, Is... > > {
                using type = list< std::integral_constant< Int, Is >... >;
            };

            template < class List >
            struct list_to_iseq;
            template < template < class... > class L, class Int, Int... Is >
            struct list_to_iseq< L< std::integral_constant< Int, Is >... > > {
                using type = gt_integer_sequence< Int, Is... >;
            };
            template < template < class... > class L >
            struct list_to_iseq< L<> > {
                using type = gt_index_sequence<>;
            };

            template < size_t N >
            using index_list = typename iseq_to_list< make_gt_index_sequence< N > >::type;

            struct any_arg {
                template < class T >
                any_arg(T &&) {}
            };
            template < class T >
            using any_arg_t = any_arg;

            template < class SomeList, class List >
            struct drop_front;
            template < template < class... > class L_, class... Us, template < class... > class L, class... Ts >
            struct drop_front< L_< Us... >, L< Ts... > > {
                template < class... Vs >
                static L< typename Vs::type... > select(any_arg_t< Us >..., Vs...);
                using type = decltype(select(id< Ts >()...));
            };

            template < class T, class Set >
            struct st_contains;
            template < class T, template < class... > class L, class... Ts >
            struct st_contains< T, L< Ts... > > {
                using type = std::is_base_of< id< T >, inherit< id< Ts >... > >;
            };

            template < class Map, class Key >
            struct mp_find;
            template < class Key, template < class... > class L, class... Ts >
            struct mp_find< L< Ts... >, Key > {
                template < template < class... > class Elem, class... Vals >
                static Elem< Key, Vals... > select(id< Elem< Key, Vals... > >);
                static void select(...);
                using type = decltype(select(std::declval< inherit< id< Ts >... > >()));
            };

            template < class T, size_t I >
            struct ipair {};

            template < class T, class Set, class ISec >
            struct st_position;
            template < class T, template < class... > class L, class... Ts, size_t... Is >
            struct st_position< T, L< Ts... >, gt_index_sequence< Is... > > {
                template < size_t I >
                static std::integral_constant< size_t, I > select(ipair< T, I >);
                static std::integral_constant< size_t, sizeof...(Ts) > select(...);
                using type = decltype(select(std::declval< inherit< ipair< Ts, Is >... > >()));
            };

            template < size_t I, class List, class ISec >
            struct at;
            template < size_t I, template < class... > class L, class... Ts, size_t... Is >
            struct at< I, L< Ts... >, gt_index_sequence< Is... > > {
                template < class T >
                static id< T > select(ipair< T, I >);
                using type = typename decltype(select(std::declval< inherit< ipair< Ts, Is >... > >()))::type;
            };

            template < template < class... > class F, class List, size_t N = length< List >::value >
            struct combine;
            template < template < class... > class F, class List, size_t N >
            struct combine {
                static_assert(N > 0, "N in combine<F, List, N> must be positive");
                static const size_t m = N / 2;
                using type = F< typename combine< F, List, m >::type,
                    typename combine< F, typename drop_front< index_list< m >, List >::type, N - m >::type >;
            };
            template < template < class... > class F, template < class... > class L, class T, class... Ts >
            struct combine< F, L< T, Ts... >, 1 > {
                using type = T;
            };
            template < template < class... > class F, template < class... > class L, class T1, class T2, class... Ts >
            struct combine< F, L< T1, T2, Ts... >, 2 > {
                using type = F< T1, T2 >;
            };
            template < template < class... > class F,
                template < class... > class L,
                class T1,
                class T2,
                class T3,
                class... Ts >
            struct combine< F, L< T1, T2, T3, Ts... >, 3 > {
                using type = F< T1, F< T2, T3 > >;
            };
            template < template < class... > class F,
                template < class... > class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts >
            struct combine< F, L< T1, T2, T3, T4, Ts... >, 4 > {
                using type = F< F< T1, T2 >, F< T3, T4 > >;
            };

            template < class... >
            struct zipper;

            template < class T, class U >
            struct zip_helper {
                using type = zipper< T, U >;
            };
            template < class T, class... Ts >
            struct zip_helper< T, zipper< Ts... > > {
                using type = zipper< T, Ts... >;
            };
            template < class T, class... Ts >
            struct zip_helper< zipper< Ts... >, T > {
                using type = zipper< Ts..., T >;
            };
            template < class... Ts, class... Us >
            struct zip_helper< zipper< Ts... >, zipper< Us... > > {
                using type = zipper< Ts..., Us... >;
            };
            template < class T, class U >
            using zip_helper_t = typename zip_helper< T, U >::type;

            template < template < class... > class F, class... Lists >
            struct transform;

            template < class... Args >
            using zip_args = typename transform< zip_helper_t, Args... >::type;

            template < template < class... > class F, class... Lists >
            struct transform
                : transform< rename< F >::template apply_t, typename combine< zip_args, list< Lists... > >::type > {};
            template < template < class... > class F, template < class... > class L, class... Ts >
            struct transform< F, L< Ts... > > {
                using type = L< F< Ts >... >;
            };
            template < template < class... > class F,
                template < class... > class L1,
                class... T1s,
                template < class... > class L2,
                class... T2s >
            struct transform< F, L1< T1s... >, L2< T2s... > > {
                using type = L1< F< T1s, T2s >... >;
            };

            template < template < class... > class F, class S, class List >
            struct rfold;
            template < template < class... > class F, class S, template < class... > class L >
            struct rfold< F, S, L<> > {
                using type = S;
            };
            template < template < class... > class F, class S, template < class... > class L, class T, class... Ts >
            struct rfold< F, S, L< T, Ts... > > {
                using type = F< T, typename rfold< F, S, L< Ts... > >::type >;
            };

            template < template < class... > class F, class S, class List >
            struct lfold;
            template < template < class... > class F, class S, template < class... > class L >
            struct lfold< F, S, L<> > {
                using type = S;
            };
            template < template < class... > class F, class S, template < class... > class L, class T, class... Ts >
            struct lfold< F, S, L< T, Ts... > > {
                using type = typename lfold< F, F< S, T >, L< Ts... > >::type;
            };

            template < class ISec, class List >
            struct make_index_map;
            template < size_t... Is, template < class... > class L, class... Ts >
            struct make_index_map< gt_index_sequence< Is... >, L< Ts... > > {
                using type = list< list< Ts, std::integral_constant< size_t, Is > >... >;
            };

            template < class List1, class List2 >
            struct concat;
            template < class List, template < class... > class L, class... Ts >
            struct concat< List, L< Ts... > > : push_back< List, Ts... > {};
            template < class... Lists >
            using concat_t = typename concat< Lists... >::type;

            template < template < class... > class Pred >
            struct filter {
                template < class... Ts >
                using apply = typename std::conditional< Pred< Ts... >::value, list< Ts... >, list<> >::type;
            };

            template < class List >
            using index_sequence_for_list = make_gt_index_sequence< length< List >::value >;

            template < class T, class S >
            using dedup_step = typename std::conditional< st_contains< T, S >::type::value,
                S,
                typename push_front< S, T >::type >::type;

            template < class List >
            struct dedup;
            template < template < class... > class L, class... Ts >
            struct dedup< L< Ts... > > {
                using type = typename rfold< dedup_step, L<>, L< Ts... > >::type;
            };
        }

        template < class List >
        using first = typename _impl::first< List >::type;

        template < class List >
        using second = typename _impl::second< List >::type;

        template < class List >
        using third = typename _impl::third< List >::type;

        template < class List, class... Ts >
        using push_front = typename _impl::push_front< List, Ts... >::type;

        template < class List, class... Ts >
        using push_back = typename _impl::push_back< List, Ts... >::type;

        /// Instantiate F with the parameters taken from List
        template < template < class... > class F, class List >
        using rename = typename _impl::rename< F >::template apply< List >::type;

        template < template < class... > class F, class List >
        using combine = typename _impl::combine< F, List >::type;

        /**
         *   Transform Lists by applying F element wise.
         *
         *   I.e the first element of resulting typelist would be F<first_from_l0, first_froml1, ...>;
         *   the second would be F<second_from_l0, ...> and so on.
         */
        template < template < class... > class F, class... Lists >
        using transform = typename _impl::transform< F, Lists... >::type;

        template < template < class U, U > class F, class ISec >
        using transform_c = typename _impl::transform_c< F, ISec >::type;

        template < class... Lists >
        using zip = transform< list, Lists... >;

        template < size_t N, class T >
        using repeat = transform< always< T >::template apply, _impl::index_list< N > >;

        template < size_t N, class T, T Val >
        using repeat_c = transform_c< always_c< T, Val >::template apply, make_gt_index_sequence< N > >;

        template < size_t N, class List >
        using drop_front = typename _impl::drop_front< _impl::index_list< N >, List >::type;

        template < template < class... > class F, class S, class List >
        using lfold = typename _impl::lfold< F, S, List >::type;

        template < template < class... > class F, class S, class List >
        using rfold = typename _impl::rfold< F, S, List >::type;

        template < size_t N, class List >
        using at = typename _impl::at< N, List, _impl::index_sequence_for_list< List > >::type;

        template < class T >
        using negation = std::integral_constant< bool, !T::value >;

        template < class... Ts >
        using conjunction = typename std::is_same< gt_integer_sequence< bool, Ts::value... >,
            repeat_c< sizeof...(Ts), bool, true > >::type;

        template < class... Ts >
        using disjunction = negation<
            std::is_same< gt_integer_sequence< bool, !Ts::value... >, repeat_c< sizeof...(Ts), bool, true > > >;

        template < class T, class Set >
        using st_contains = typename _impl::st_contains< T, Set >::type;

        template < class Map, class Key >
        using mp_find = typename _impl::mp_find< Map, Key >::type;

        template < class Set >
        using st_make_index_map = typename _impl::make_index_map< _impl::index_sequence_for_list< Set >, Set >::type;

        template < class T, class Set >
        using st_position = typename _impl::st_position< T, Set, _impl::index_sequence_for_list< Set > >::type;

        template < class List >
        using flatten = combine< _impl::concat_t, List >;

        template < class... Lists >
        using concat = flatten< list< Lists... > >;

        template < template < class... > class Pred, class List >
        using filter = flatten< transform< _impl::filter< Pred >::template apply, List > >;

        template < class List >
        using dedup = typename _impl::dedup< List >::type;

        template < template < class... > class F, class T >
        struct bind_second {
            template < class A >
            using apply = F< A, T >;
        };

        template < class List, class Set >
        using st_positions =
            typename _impl::list_to_iseq< transform< bind_second< st_position, Set >::template apply, List > >::type;

        // TODO(anstaf): Add is_set<List>, st_equiv<List1, List2>, is_map<List>

        template < class List >
        struct all;
        template < template < class... > class L, class... Ts >
        struct all< L< Ts... > > : conjunction< Ts... > {};

        template < class List >
        struct any;
        template < template < class... > class L, class... Ts >
        struct any< L< Ts... > > : disjunction< Ts... > {};

        template < template < class... > class Pred, class List >
        using all_of = all< transform< Pred, List > >;

        template < template < class... > class Pred, class List >
        using any_of = any< transform< Pred, List > >;

        template < template < class... > class L >
        struct is_instantiation_of {
            template < class T >
            struct apply : std::false_type {};
            template < class... Ts >
            struct apply< L< Ts... > > : std::true_type {};
        };
    }
}
