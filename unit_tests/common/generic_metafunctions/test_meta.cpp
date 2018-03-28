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

#include <common/generic_metafunctions/meta.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace gridtools {
    namespace meta {
        template < class... >
        struct f;
        template < class... >
        struct g;

        // is_list
        static_assert(!is_list< int >{}, "");
        static_assert(is_list< list< int, void > >{}, "");
        static_assert(is_list< f< void, int > >{}, "");
        static_assert(is_list< std::pair< int, double > >{}, "");
        static_assert(is_list< std::tuple< int, double > >{}, "");

        // has_type
        static_assert(!has_type< int >{}, "");
        static_assert(!has_type< f<> >{}, "");
        static_assert(has_type< lazy::id< void > >{}, "");
        static_assert(has_type< std::is_void< int > >{}, "");

        // is_meta_class
        static_assert(!is_meta_class< int >{}, "");
        static_assert(!is_meta_class< f< int > >{}, "");
        static_assert(!is_meta_class< std::is_void< int > >{}, "");
        static_assert(is_meta_class< always< int > >{}, "");
        static_assert(is_meta_class< curry< f > >{}, "");
        static_assert(is_meta_class< ctor< f<> > >{}, "");

        // length
        static_assert(length< list<> >::value == 0, "");
        static_assert(length< std::tuple< int > >::value == 1, "");
        static_assert(length< std::pair< int, double > >::value == 2, "");
        static_assert(length< f< int, int, double > >::value == 3, "");

        // ctor
        static_assert(std::is_same< GT_META_CALL(ctor< f< double > >::apply, (int, void)), f< int, void > >{}, "");

        // rename
        static_assert(
            std::is_same< GT_META_CALL(rename, (GT_META_DIRECT_PARAM(f), g< int, double >)), f< int, double > >{}, "");

        // transform
        static_assert(std::is_same< GT_META_CALL(transform, (GT_META_DIRECT_PARAM(f), g<>)), g<> >{}, "");
        static_assert(std::is_same< GT_META_CALL(transform, (GT_META_DIRECT_PARAM(f), g< int, void >)),
                          g< f< int >, f< void > > >{},
            "");
        static_assert(
            std::is_same< GT_META_CALL(transform,
                              (GT_META_DIRECT_PARAM(f), g< int, void >, g< int *, void * >, g< int **, void ** >)),
                g< f< int, int *, int ** >, f< void, void *, void ** > > >{},
            "");

        // st_contains
        static_assert(st_contains< g< int, bool >, int >{}, "");
        static_assert(!st_contains< g< int, bool >, void >{}, "");

        // mp_find
        using map = f< g< int, void * >, g< void, double * >, g< float, double * > >;
        static_assert(std::is_same< GT_META_CALL(mp_find, (map, int)), g< int, void * > >{}, "");
        static_assert(std::is_same< GT_META_CALL(mp_find, (map, double)), void >{}, "");

        // repeat
        static_assert(std::is_same< GT_META_CALL(repeat, (0, int)), list<> >{}, "");
        static_assert(std::is_same< GT_META_CALL(repeat, (3, int)), list< int, int, int > >{}, "");

        // drop_front
        static_assert(std::is_same< GT_META_CALL(drop_front_c, (0, f< int, double >)), f< int, double > >{}, "");
        static_assert(std::is_same< GT_META_CALL(drop_front_c, (1, f< int, double >)), f< double > >{}, "");
        static_assert(std::is_same< GT_META_CALL(drop_front_c, (2, f< int, double >)), f<> >{}, "");

        // at
        static_assert(std::is_same< GT_META_CALL(at_c, (f< int, double >, 0)), int >{}, "");
        static_assert(std::is_same< GT_META_CALL(at_c, (f< int, double >, 1)), double >{}, "");

        // conjunction
        static_assert(conjunction_fast<>{}, "");
        static_assert(conjunction_fast< std::true_type, std::true_type >{}, "");
        static_assert(!conjunction_fast< std::true_type, std::false_type >{}, "");
        static_assert(!conjunction_fast< std::false_type, std::true_type >{}, "");
        static_assert(!conjunction_fast< std::false_type, std::false_type >{}, "");

        // disjunction
        static_assert(!disjunction_fast<>{}, "");
        static_assert(disjunction_fast< std::true_type, std::true_type >{}, "");
        static_assert(disjunction_fast< std::true_type, std::false_type >{}, "");
        static_assert(disjunction_fast< std::false_type, std::true_type >{}, "");
        static_assert(!disjunction_fast< std::false_type, std::false_type >{}, "");

        // st_position
        static_assert(st_position< f< int, double >, int >{} == 0, "");
        static_assert(st_position< f< double, int >, int >{} == 1, "");
        static_assert(st_position< f< double, int >, void >{} == 2, "");

        // combine
        static_assert(std::is_same< GT_META_CALL(combine, (GT_META_DIRECT_PARAM(f), g< int >)), int >{}, "");
        static_assert(std::is_same< GT_META_CALL(combine, (GT_META_DIRECT_PARAM(f), GT_META_CALL(repeat, (8, int)))),
                          f< f< f< int, int >, f< int, int > >, f< f< int, int >, f< int, int > > > >{},
            "");

        // concat
        static_assert(std::is_same< GT_META_CALL(concat, g< int >), g< int > >{}, "");
        static_assert(std::is_same< GT_META_CALL(concat, (g< int >, f< void >)), g< int, void > >{}, "");
        static_assert(std::is_same< GT_META_CALL(concat, (g< int >, g< void, double >, g< void, int >)),
                          g< int, void, double, void, int > >{},
            "");

        // filter
        static_assert(std::is_same< GT_META_CALL(filter, (std::is_pointer, f<>)), f<> >{}, "");
        static_assert(std::is_same< GT_META_CALL(filter, (std::is_pointer, f< void, int *, double, double ** >)),
                          f< int *, double ** > >{},
            "");

        // all_of
        static_assert(all_of< is_list, f< f<>, f< int > > >{}, "");

        // dedup
        static_assert(std::is_same< GT_META_CALL(dedup, f<>), f<> >{}, "");
        static_assert(std::is_same< GT_META_CALL(dedup, f< int >), f< int > >{}, "");
        static_assert(std::is_same< GT_META_CALL(dedup, (f< int, void >)), f< int, void > >{}, "");
        static_assert(
            std::is_same< GT_META_CALL(dedup, (f< int, void, void, void, int, void >)), f< int, void > >{}, "");

        // zip
        static_assert(std::is_same< GT_META_CALL(zip, (f< int >, f< void >)), f< list< int, void > > >{}, "");
        static_assert(
            std::is_same< GT_META_CALL(
                              zip, (f< int, int *, int ** >, f< void, void *, void ** >, f< char, char *, char ** >)),
                f< list< int, void, char >, list< int *, void *, char * >, list< int **, void **, char ** > > >{},
            "");

        // bind
        static_assert(std::is_same< GT_META_CALL((bind< GT_META_DIRECT_PARAM(f), _2, void, _1 >::apply), (int, double)),
                          f< double, void, int > >{},
            "");

        // is_instantiation_of
        static_assert(is_instantiation_of< f, f<> >{}, "");
        static_assert(is_instantiation_of< f, f< int, void > >{}, "");
        static_assert(!is_instantiation_of< f, g<> >{}, "");
        static_assert(!is_instantiation_of< f, int >{}, "");

        static_assert(std::is_same< GT_META_CALL(replace, (f< int, double, int, double >, double, void)),
                          f< int, void, int, void > >{},
            "");

        static_assert(std::is_same< GT_META_CALL(mp_replace, (f< g< int, int * >, g< double, double * > >, int, void)),
                          f< g< int, void >, g< double, double * > > >{},
            "");

        static_assert(std::is_same< GT_META_CALL(replace_at_c, (f< int, double, int, double >, 1, void)),
                          f< int, void, int, double > >{},
            "");

        namespace nvcc_sizeof_workaround {
            template < class... >
            struct a;

            template < int I >
            struct b {
                using c = void;
            };

            template < class... Ts >
            using d = b< GT_SIZEOF_3_DOTS(Ts) >;

            template < class... Ts >
            using e = typename d< a< Ts >... >::c;
        }

        static_assert(is_set< f<> >{}, "");
        static_assert(is_set< f< int > >{}, "");
        static_assert(is_set< f< void > >{}, "");
        static_assert(is_set< f< int, void > >{}, "");
        static_assert(!is_set< int >{}, "");
        static_assert(!is_set< f< int, void, int > >{}, "");

        static_assert(is_set_fast< f<> >{}, "");
        static_assert(is_set_fast< f< int > >{}, "");
        static_assert(is_set_fast< f< void > >{}, "");
        static_assert(is_set_fast< f< int, void > >{}, "");
        static_assert(!is_set_fast< int >{}, "");
        //        static_assert(!is_set_fast< f< int, void, int > >{}, "");
    }
}

TEST(dummy, dummy) {}
