/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/meta.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace gridtools {
    namespace meta {
        template <class...>
        struct f;
        template <class...>
        struct g;
        template <class...>
        struct h;

        // is_list
        static_assert(!is_list<int>{}, "");
        static_assert(is_list<list<int, void>>{}, "");
        static_assert(is_list<f<void, int>>{}, "");
        static_assert(is_list<std::pair<int, double>>{}, "");
        static_assert(is_list<std::tuple<int, double>>{}, "");

        // has_type
        static_assert(!has_type<int>{}, "");
        static_assert(!has_type<g<>>{}, "");
        static_assert(has_type<lazy::id<void>>{}, "");
        static_assert(has_type<std::is_void<int>>{}, "");

        // is_meta_class
        static_assert(!is_meta_class<int>{}, "");
        static_assert(!is_meta_class<f<int>>{}, "");
        static_assert(!is_meta_class<std::is_void<int>>{}, "");
        static_assert(is_meta_class<always<int>>{}, "");
        static_assert(is_meta_class<curry<f>>{}, "");
        static_assert(is_meta_class<ctor<f<>>>{}, "");

        // length
        static_assert(length<list<>>::value == 0, "");
        static_assert(length<std::tuple<int>>::value == 1, "");
        static_assert(length<std::pair<int, double>>::value == 2, "");
        static_assert(length<f<int, int, double>>::value == 3, "");

        // ctor
        static_assert(std::is_same<ctor<f<double>>::apply<int, void>, f<int, void>>{}, "");

        // rename
        static_assert(std::is_same<rename<f, g<int, double>>, f<int, double>>{}, "");

        // transform
        static_assert(std::is_same<transform<f, g<>>, g<>>{}, "");
        static_assert(std::is_same<transform<f, g<int, void>>, g<f<int>, f<void>>>{}, "");
        static_assert(std::is_same<transform<f, g<int, void>, g<int *, void *>, g<int **, void **>>,
                          g<f<int, int *, int **>, f<void, void *, void **>>>{},
            "");

        // st_contains
        static_assert(st_contains<g<int, bool>, int>{}, "");
        static_assert(!st_contains<g<int, bool>, void>{}, "");

        // mp_find
        using map = f<g<int, void *>, g<void, double *>, g<float, double *>>;
        static_assert(std::is_same<mp_find<map, int>, g<int, void *>>{}, "");
        static_assert(std::is_same<mp_find<map, double>, void>{}, "");

        // repeat
        static_assert(std::is_same<repeat_c<0, int>, list<>>{}, "");
        static_assert(std::is_same<repeat_c<3, int>, list<int, int, int>>{}, "");

        // drop_front
        static_assert(std::is_same<drop_front_c<0, f<int, double>>, f<int, double>>{}, "");
        static_assert(std::is_same<drop_front_c<1, f<int, double>>, f<double>>{}, "");
        static_assert(std::is_same<drop_front_c<2, f<int, double>>, f<>>{}, "");

        // at
        static_assert(std::is_same<at_c<f<int, double>, 0>, int>{}, "");
        static_assert(std::is_same<at_c<f<int, double>, 1>, double>{}, "");
        static_assert(std::is_same<last<f<int, double>>, double>{}, "");

        // conjunction
        static_assert(conjunction_fast<>{}, "");
        static_assert(conjunction_fast<std::true_type, std::true_type>{}, "");
        static_assert(!conjunction_fast<std::true_type, std::false_type>{}, "");
        static_assert(!conjunction_fast<std::false_type, std::true_type>{}, "");
        static_assert(!conjunction_fast<std::false_type, std::false_type>{}, "");

        // disjunction
        static_assert(!disjunction_fast<>{}, "");
        static_assert(disjunction_fast<std::true_type, std::true_type>{}, "");
        static_assert(disjunction_fast<std::true_type, std::false_type>{}, "");
        static_assert(disjunction_fast<std::false_type, std::true_type>{}, "");
        static_assert(!disjunction_fast<std::false_type, std::false_type>{}, "");

        // st_position
        static_assert(st_position<f<int, double>, int>{} == 0, "");
        static_assert(st_position<f<double, int>, int>{} == 1, "");
        static_assert(st_position<f<double, int>, void>{} == 2, "");

        // combine
        static_assert(std::is_same<combine<f, g<int>>, int>{}, "");
        static_assert(
            std::is_same<combine<f, repeat_c<8, int>>, f<f<f<int, int>, f<int, int>>, f<f<int, int>, f<int, int>>>>{},
            "");
        static_assert(std::is_same<combine<f, g<int, int>>, f<int, int>>{}, "");
        static_assert(std::is_same<combine<f, g<int, int, int>>, f<int, f<int, int>>>{}, "");
        static_assert(std::is_same<combine<f, repeat_c<4, int>>, f<f<int, int>, f<int, int>>>{}, "");

        // concat
        static_assert(std::is_same<concat<g<int>>, g<int>>{}, "");
        static_assert(std::is_same<concat<g<int>, f<void>>, g<int, void>>{}, "");
        static_assert(
            std::is_same<concat<g<int>, g<void, double>, g<void, int>>, g<int, void, double, void, int>>{}, "");

        // flatten
        static_assert(std::is_same<flatten<f<g<int>>>, g<int>>{}, "");
        static_assert(std::is_same<flatten<f<g<int>, f<bool>>>, g<int, bool>>{}, "");

        // filter
        static_assert(std::is_same<filter<std::is_pointer, f<>>, f<>>{}, "");
        static_assert(
            std::is_same<filter<std::is_pointer, f<void, int *, double, double **>>, f<int *, double **>>{}, "");

        // all_of
        static_assert(all_of<is_list, f<f<>, f<int>>>{}, "");

        // dedup
        static_assert(std::is_same<dedup<f<>>, f<>>{}, "");
        static_assert(std::is_same<dedup<f<int>>, f<int>>{}, "");
        static_assert(std::is_same<dedup<f<int, void>>, f<int, void>>{}, "");
        static_assert(std::is_same<dedup<f<int, void, void, void, int, void>>, f<int, void>>{}, "");

        // zip
        static_assert(std::is_same<zip<f<int>, f<void>>, f<list<int, void>>>{}, "");
        static_assert(std::is_same<zip<f<int, int *, int **>, f<void, void *, void **>, f<char, char *, char **>>,
                          f<list<int, void, char>, list<int *, void *, char *>, list<int **, void **, char **>>>{},
            "");

        // bind
        static_assert(std::is_same<bind<f, _2, void, _1>::apply<int, double>, f<double, void, int>>{}, "");

        // is_instantiation_of
        static_assert(is_instantiation_of<f, f<>>{}, "");
        static_assert(is_instantiation_of<f, f<int, void>>{}, "");
        static_assert(!is_instantiation_of<f, g<>>{}, "");
        static_assert(!is_instantiation_of<f, int>{}, "");
        static_assert(is_instantiation_of<f>::apply<f<int, void>>{}, "");

        static_assert(std::is_same<replace<f<int, double, int, double>, double, void>, f<int, void, int, void>>{}, "");

        static_assert(std::is_same<mp_replace<f<g<int, int *>, g<double, double *>>, int, void>,
                          f<g<int, void>, g<double, double *>>>{},
            "");

        static_assert(
            std::is_same<replace_at_c<f<int, double, int, double>, 1, void>, f<int, void, int, double>>{}, "");

        namespace nvcc_sizeof_workaround {
            template <class...>
            struct a;

            template <int I>
            struct b {
                using c = void;
            };

            template <class... Ts>
            using d = b<GT_SIZEOF_3_DOTS(Ts)>;

            template <class... Ts>
            using e = typename d<a<Ts>...>::c;
        } // namespace nvcc_sizeof_workaround

        static_assert(is_set<f<>>{}, "");
        static_assert(is_set<f<int>>{}, "");
        static_assert(is_set<f<void>>{}, "");
        static_assert(is_set<f<int, void>>{}, "");
        static_assert(!is_set<int>{}, "");
        static_assert(!is_set<f<int, void, int>>{}, "");

        static_assert(is_set_fast<f<>>{}, "");
        static_assert(is_set_fast<f<int>>{}, "");
        static_assert(is_set_fast<f<void>>{}, "");
        static_assert(is_set_fast<f<int, void>>{}, "");
        static_assert(!is_set_fast<int>{}, "");
        //        static_assert(!is_set_fast< f< int, void, int > >{}, "");

        // lfold
        static_assert(std::is_same<lfold<f, int, g<>>, int>{}, "");
        static_assert(std::is_same<lfold<f, int, g<int>>, f<int, int>>{}, "");
        static_assert(std::is_same<lfold<f, int, g<int, int>>, f<f<int, int>, int>>{}, "");
        static_assert(std::is_same<lfold<f, int, g<int, int>>, f<f<int, int>, int>>{}, "");
        static_assert(std::is_same<lfold<f, int, g<int, int, int>>, f<f<f<int, int>, int>, int>>{}, "");
        static_assert(std::is_same<lfold<f, int, g<int, int, int, int>>, f<f<f<f<int, int>, int>, int>, int>>{}, "");
        static_assert(
            std::is_same<lfold<f, int, g<int, int, int, int, int>>, f<f<f<f<f<int, int>, int>, int>, int>, int>>{}, "");

        // rfold
        static_assert(std::is_same<rfold<f, int, g<>>, int>{}, "");
        static_assert(std::is_same<rfold<f, int, g<int>>, f<int, int>>{}, "");
        static_assert(std::is_same<rfold<f, int, g<int, int>>, f<int, f<int, int>>>{}, "");
        static_assert(std::is_same<rfold<f, int, g<int, int, int>>, f<int, f<int, f<int, int>>>>{}, "");
        static_assert(std::is_same<rfold<f, int, g<int, int, int, int>>, f<int, f<int, f<int, f<int, int>>>>>{}, "");
        static_assert(
            std::is_same<rfold<f, int, g<int, int, int, int, int>>, f<int, f<int, f<int, f<int, f<int, int>>>>>>{}, "");

        static_assert(std::is_same<cartesian_product<>, list<list<>>>{}, "");
        static_assert(std::is_same<cartesian_product<f<>>, list<>>{}, "");
        static_assert(std::is_same<cartesian_product<f<int>>, list<list<int>>>{}, "");
        static_assert(std::is_same<cartesian_product<f<int, double>>, list<list<int>, list<double>>>{}, "");
        static_assert(
            std::is_same<cartesian_product<f<int, double>, g<void>>, list<list<int, void>, list<double, void>>>{}, "");
        static_assert(std::is_same<cartesian_product<f<int, double>, g<int *, double *>>,
                          list<list<int, int *>, list<int, double *>, list<double, int *>, list<double, double *>>>{},
            "");
        static_assert(std::is_same<cartesian_product<f<int, double>, g<>, f<void>>, list<>>{}, "");
        static_assert(std::is_same<cartesian_product<f<>, g<int, double>>, list<>>{}, "");
        static_assert(
            std::is_same<cartesian_product<f<int>, g<double>, list<void>>, list<list<int, double, void>>>{}, "");

        static_assert(std::is_same<reverse<f<int, int *, int **, int ***, int ****, int *****>>,
                          f<int *****, int ****, int ***, int **, int *, int>>{},
            "");

        static_assert(find<f<>, int>::type::value == 0, "");
        static_assert(find<f<void>, int>::type::value == 1, "");
        static_assert(find<f<double, int, int, double, int>, int>::type::value == 1, "");
        static_assert(find<f<double, int, int, double, int>, void>::type::value == 5, "");

        static_assert(std::is_same<mp_insert<f<>, g<int, int *>>, f<g<int, int *>>>{}, "");
        static_assert(
            std::is_same<mp_insert<f<g<void, void *>>, g<int, int *>>, f<g<void, void *>, g<int, int *>>>{}, "");
        static_assert(std::is_same<mp_insert<f<g<int, int *>>, g<int, int **>>, f<g<int, int *, int **>>>{}, "");

        static_assert(std::is_same<mp_remove<f<g<int, int *>>, void>, f<g<int, int *>>>{}, "");
        static_assert(std::is_same<mp_remove<f<g<int, int *>>, int>, f<>>{}, "");
        static_assert(std::is_same<mp_remove<f<g<int, int *>, g<void, void *>>, int>, f<g<void, void *>>>{}, "");

        static_assert(std::is_same<mp_inverse<f<>>, f<>>{}, "");
        static_assert(
            std::is_same<mp_inverse<f<g<int, int *>, g<void, void *>>>, f<g<int *, int>, g<void *, void>>>{}, "");
        static_assert(std::is_same<mp_inverse<f<g<int, int *, int **>, g<void, void *>>>,
                          f<g<int *, int>, g<int **, int>, g<void *, void>>>{},
            "");
        static_assert(std::is_same<mp_inverse<f<g<int *, int>, g<int **, int>, g<void *, void>>>,
                          f<g<int, int *, int **>, g<void, void *>>>{},
            "");

        // take
        static_assert(std::is_same<take_c<2, f<int, double, void, void>>, f<int, double>>::value, "");
        static_assert(std::is_same<take_c<20, repeat_c<100, int>>, repeat_c<20, int>>::value, "");

        // insert
        static_assert(std::is_same<insert_c<3, f<void, void, void, void, void>, int, double>,
                          f<void, void, void, int, double, void, void>>::value,
            "");

        // void_t (CWG 1558 https://wg21.cmeerw.net/cwg/issue1558)
        namespace defect_cwg_1558 {
            template <class, class = void_t<>>
            struct has_type_member : std::false_type {};

            // specialization recognizes types that do have a nested ::type member:
            template <class T>
            struct has_type_member<T, void_t<typename T::type>> : std::true_type {};

            struct X {
                using type = void;
            };

            static_assert(!has_type_member<int>::value, "");
            static_assert(has_type_member<X>::value, "");
        } // namespace defect_cwg_1558

        // group
        static_assert(std::is_same<group<are_same, g, f<>>, f<>>::value, "");
        static_assert(std::is_same<group<are_same, g, f<int>>, f<g<int>>>::value, "");
        static_assert(std::is_same<group<are_same, g, f<int, int, int, double, void, void, int, int>>,
                          f<g<int, int, int>, g<double>, g<void, void>, g<int, int>>>::value,
            "");

        // trim
        static_assert(std::is_same<trim<std::is_void, f<int, void, int>>, f<int, void, int>>::value, "");
        static_assert(std::is_same<trim<std::is_void, f<>>, f<>>::value, "");
        static_assert(std::is_same<trim<std::is_void, f<void, void>>, f<>>::value, "");
        static_assert(
            std::is_same<trim<std::is_void, f<void, void, int, int, void, int, void>>, f<int, int, void, int>>::value,
            "");

        // mp_merge
        static_assert(std::is_same<mp_merge<h>, list<>>::value, "");
        static_assert(std::is_same<mp_merge<h, f<g<int, int *>>>, f<h<g<int, int *>>>>::value, "");
        static_assert(std::is_same<mp_merge<h, f<g<int, int *>, g<void, void *>>>,
                          f<h<g<int, int *>>, h<g<void, void *>>>>::value,
            "");
        static_assert(std::is_same<mp_merge<h, f<g<int, int *>>, f<g<void, void *>>>,
                          f<h<g<int, int *>>, h<g<void, void *>>>>::value,
            "");
        static_assert(
            std::is_same<mp_merge<h, f<g<int, int *>>, f<g<int, int **>>>, f<h<g<int, int *>, g<int, int **>>>>::value,
            "");
        static_assert(
            std::is_same<mp_merge<h, f<g<void, void *>, g<int, int *>>, f<g<int, int **>, g<double, double **>>>,
                f<h<g<void, void *>>, h<g<int, int *>, g<int, int **>>, h<g<double, double **>>>>::value,
            "");
    } // namespace meta
} // namespace gridtools

TEST(dummy, dummy) {}
