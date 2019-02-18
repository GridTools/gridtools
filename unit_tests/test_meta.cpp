/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
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
#if GT_BROKEN_TEMPLATE_ALIASES
        template <class...>
        struct f {
            using type = f;
        };
#endif
        template <class...>
        struct g;

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
        static_assert(std::is_same<GT_META_CALL(ctor<f<double>>::apply, (int, void)), f<int, void>>{}, "");

        // rename
        static_assert(std::is_same<GT_META_CALL(rename, (f, g<int, double>)), f<int, double>>{}, "");

        // transform
        static_assert(std::is_same<GT_META_CALL(transform, (f, g<>)), g<>>{}, "");
        static_assert(std::is_same<GT_META_CALL(transform, (f, g<int, void>)), g<f<int>, f<void>>>{}, "");
        static_assert(std::is_same<GT_META_CALL(transform, (f, g<int, void>, g<int *, void *>, g<int **, void **>)),
                          g<f<int, int *, int **>, f<void, void *, void **>>>{},
            "");

        // st_contains
        static_assert(st_contains<g<int, bool>, int>{}, "");
        static_assert(!st_contains<g<int, bool>, void>{}, "");

        // mp_find
        using map = f<g<int, void *>, g<void, double *>, g<float, double *>>;
        static_assert(std::is_same<GT_META_CALL(mp_find, (map, int)), g<int, void *>>{}, "");
        static_assert(std::is_same<GT_META_CALL(mp_find, (map, double)), void>{}, "");

        // repeat
        static_assert(std::is_same<GT_META_CALL(repeat_c, (0, int)), list<>>{}, "");
        static_assert(std::is_same<GT_META_CALL(repeat_c, (3, int)), list<int, int, int>>{}, "");

        // drop_front
        static_assert(std::is_same<GT_META_CALL(drop_front_c, (0, f<int, double>)), f<int, double>>{}, "");
        static_assert(std::is_same<GT_META_CALL(drop_front_c, (1, f<int, double>)), f<double>>{}, "");
        static_assert(std::is_same<GT_META_CALL(drop_front_c, (2, f<int, double>)), f<>>{}, "");

        // at
        static_assert(std::is_same<GT_META_CALL(at_c, (f<int, double>, 0)), int>{}, "");
        static_assert(std::is_same<GT_META_CALL(at_c, (f<int, double>, 1)), double>{}, "");
        static_assert(std::is_same<GT_META_CALL(last, (f<int, double>)), double>{}, "");

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
        static_assert(std::is_same<GT_META_CALL(combine, (f, g<int>)), int>{}, "");
        static_assert(std::is_same<GT_META_CALL(combine, (f, GT_META_CALL(repeat_c, (8, int)))),
                          f<f<f<int, int>, f<int, int>>, f<f<int, int>, f<int, int>>>>{},
            "");
        static_assert(std::is_same<GT_META_CALL(combine, (f, g<int, int>)), f<int, int>>{}, "");
        static_assert(std::is_same<GT_META_CALL(combine, (f, g<int, int, int>)), f<int, f<int, int>>>{}, "");
        static_assert(
            std::is_same<GT_META_CALL(combine, (f, GT_META_CALL(repeat_c, (4, int)))), f<f<int, int>, f<int, int>>>{},
            "");

        // concat
        static_assert(std::is_same<GT_META_CALL(concat, g<int>), g<int>>{}, "");
        static_assert(std::is_same<GT_META_CALL(concat, (g<int>, f<void>)), g<int, void>>{}, "");
        static_assert(std::is_same<GT_META_CALL(concat, (g<int>, g<void, double>, g<void, int>)),
                          g<int, void, double, void, int>>{},
            "");

        // flatten
        static_assert(std::is_same<GT_META_CALL(flatten, f<g<int>>), g<int>>{}, "");
        static_assert(std::is_same<GT_META_CALL(flatten, (f<g<int>, f<bool>>)), g<int, bool>>{}, "");

        // filter
        static_assert(std::is_same<GT_META_CALL(filter, (std::is_pointer, f<>)), f<>>{}, "");
        static_assert(std::is_same<GT_META_CALL(filter, (std::is_pointer, f<void, int *, double, double **>)),
                          f<int *, double **>>{},
            "");

        // all_of
        static_assert(all_of<is_list, f<f<>, f<int>>>{}, "");

        // dedup
        static_assert(std::is_same<GT_META_CALL(dedup, f<>), f<>>{}, "");
        static_assert(std::is_same<GT_META_CALL(dedup, f<int>), f<int>>{}, "");
        static_assert(std::is_same<GT_META_CALL(dedup, (f<int, void>)), f<int, void>>{}, "");
        static_assert(std::is_same<GT_META_CALL(dedup, (f<int, void, void, void, int, void>)), f<int, void>>{}, "");

        // zip
        static_assert(std::is_same<GT_META_CALL(zip, (f<int>, f<void>)), f<list<int, void>>>{}, "");
        static_assert(
            std::is_same<GT_META_CALL(zip, (f<int, int *, int **>, f<void, void *, void **>, f<char, char *, char **>)),
                f<list<int, void, char>, list<int *, void *, char *>, list<int **, void **, char **>>>{},
            "");

        // bind
        static_assert(
            std::is_same<GT_META_CALL((bind<f, _2, void, _1>::apply), (int, double)), f<double, void, int>>{}, "");

        // is_instantiation_of
        static_assert(is_instantiation_of<f, f<>>{}, "");
        static_assert(is_instantiation_of<f, f<int, void>>{}, "");
        static_assert(!is_instantiation_of<f, g<>>{}, "");
        static_assert(!is_instantiation_of<f, int>{}, "");
        static_assert(is_instantiation_of<f>::apply<f<int, void>>{}, "");

        static_assert(
            std::is_same<GT_META_CALL(replace, (f<int, double, int, double>, double, void)), f<int, void, int, void>>{},
            "");

        static_assert(std::is_same<GT_META_CALL(mp_replace, (f<g<int, int *>, g<double, double *>>, int, void)),
                          f<g<int, void>, g<double, double *>>>{},
            "");

        static_assert(std::is_same<GT_META_CALL(replace_at_c, (f<int, double, int, double>, 1, void)),
                          f<int, void, int, double>>{},
            "");

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
        static_assert(std::is_same<GT_META_CALL(lfold, (f, int, g<>)), int>{}, "");
        static_assert(std::is_same<GT_META_CALL(lfold, (f, int, g<int>)), f<int, int>>{}, "");
        static_assert(std::is_same<GT_META_CALL(lfold, (f, int, g<int, int>)), f<f<int, int>, int>>{}, "");
        static_assert(std::is_same<GT_META_CALL(lfold, (f, int, g<int, int>)), f<f<int, int>, int>>{}, "");
        static_assert(std::is_same<GT_META_CALL(lfold, (f, int, g<int, int, int>)), f<f<f<int, int>, int>, int>>{}, "");
        static_assert(
            std::is_same<GT_META_CALL(lfold, (f, int, g<int, int, int, int>)), f<f<f<f<int, int>, int>, int>, int>>{},
            "");
        static_assert(std::is_same<GT_META_CALL(lfold, (f, int, g<int, int, int, int, int>)),
                          f<f<f<f<f<int, int>, int>, int>, int>, int>>{},
            "");

        // rfold
        static_assert(std::is_same<GT_META_CALL(rfold, (f, int, g<>)), int>{}, "");
        static_assert(std::is_same<GT_META_CALL(rfold, (f, int, g<int>)), f<int, int>>{}, "");
        static_assert(std::is_same<GT_META_CALL(rfold, (f, int, g<int, int>)), f<int, f<int, int>>>{}, "");
        static_assert(std::is_same<GT_META_CALL(rfold, (f, int, g<int, int, int>)), f<int, f<int, f<int, int>>>>{}, "");
        static_assert(
            std::is_same<GT_META_CALL(rfold, (f, int, g<int, int, int, int>)), f<int, f<int, f<int, f<int, int>>>>>{},
            "");
        static_assert(std::is_same<GT_META_CALL(rfold, (f, int, g<int, int, int, int, int>)),
                          f<int, f<int, f<int, f<int, f<int, int>>>>>>{},
            "");

        static_assert(std::is_same<GT_META_CALL(cartesian_product, ()), list<list<>>>{}, "");
        static_assert(std::is_same<GT_META_CALL(cartesian_product, (f<>)), list<>>{}, "");
        static_assert(std::is_same<GT_META_CALL(cartesian_product, (f<int>)), list<list<int>>>{}, "");
        static_assert(
            std::is_same<GT_META_CALL(cartesian_product, (f<int, double>)), list<list<int>, list<double>>>{}, "");
        static_assert(std::is_same<GT_META_CALL(cartesian_product, (f<int, double>, g<void>)),
                          list<list<int, void>, list<double, void>>>{},
            "");
        static_assert(std::is_same<GT_META_CALL(cartesian_product, (f<int, double>, g<int *, double *>)),
                          list<list<int, int *>, list<int, double *>, list<double, int *>, list<double, double *>>>{},
            "");
        static_assert(std::is_same<GT_META_CALL(cartesian_product, (f<int, double>, g<>, f<void>)), list<>>{}, "");
        static_assert(std::is_same<GT_META_CALL(cartesian_product, (f<>, g<int, double>)), list<>>{}, "");
        static_assert(std::is_same<GT_META_CALL(cartesian_product, (f<int>, g<double>, list<void>)),
                          list<list<int, double, void>>>{},
            "");

        static_assert(std::is_same<GT_META_CALL(reverse, (f<int, int *, int **, int ***, int ****, int *****>)),
                          f<int *****, int ****, int ***, int **, int *, int>>{},
            "");

        static_assert(find<f<>, int>::type::value == 0, "");
        static_assert(find<f<void>, int>::type::value == 1, "");
        static_assert(find<f<double, int, int, double, int>, int>::type::value == 1, "");
        static_assert(find<f<double, int, int, double, int>, void>::type::value == 5, "");

        static_assert(std::is_same<GT_META_CALL(mp_insert, (f<>, g<int, int *>)), f<g<int, int *>>>{}, "");
        static_assert(std::is_same<GT_META_CALL(mp_insert, (f<g<void, void *>>, g<int, int *>)),
                          f<g<void, void *>, g<int, int *>>>{},
            "");
        static_assert(
            std::is_same<GT_META_CALL(mp_insert, (f<g<int, int *>>, g<int, int **>)), f<g<int, int *, int **>>>{}, "");

        static_assert(std::is_same<GT_META_CALL(mp_remove, (f<g<int, int *>>, void)), f<g<int, int *>>>{}, "");
        static_assert(std::is_same<GT_META_CALL(mp_remove, (f<g<int, int *>>, int)), f<>>{}, "");
        static_assert(
            std::is_same<GT_META_CALL(mp_remove, (f<g<int, int *>, g<void, void *>>, int)), f<g<void, void *>>>{}, "");

        static_assert(std::is_same<GT_META_CALL(mp_inverse, f<>), f<>>{}, "");
        static_assert(std::is_same<GT_META_CALL(mp_inverse, (f<g<int, int *>, g<void, void *>>)),
                          f<g<int *, int>, g<void *, void>>>{},
            "");
        static_assert(std::is_same<GT_META_CALL(mp_inverse, (f<g<int, int *, int **>, g<void, void *>>)),
                          f<g<int *, int>, g<int **, int>, g<void *, void>>>{},
            "");
        static_assert(std::is_same<GT_META_CALL(mp_inverse, (f<g<int *, int>, g<int **, int>, g<void *, void>>)),
                          f<g<int, int *, int **>, g<void, void *>>>{},
            "");

        static_assert(std::is_same<integer_sequence<int>::value_type, int>::value, "");
        static_assert(integer_sequence<int, 1, 2, 3>::size() == 3, "");

        static_assert(std::is_same<make_integer_sequence<int, 3>, integer_sequence<int, 0, 1, 2>>::value, "");
        static_assert(std::is_same<make_integer_sequence<bool, 1>, integer_sequence<bool, false>>::value, "");
    } // namespace meta
} // namespace gridtools

TEST(dummy, dummy) {}
