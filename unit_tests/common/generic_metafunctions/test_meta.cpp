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
        class a_list {};

        static_assert(!is_list< int >{}, "");
        static_assert(is_list< list< int, void > >{}, "");
        static_assert(is_list< a_list< void, int > >{}, "");
        static_assert(is_list< std::pair< int, double > >{}, "");
        static_assert(is_list< std::tuple< int, double > >{}, "");

        static_assert(length< list<> >::value == 0, "");
        static_assert(length< std::tuple< int > >::value == 1, "");
        static_assert(length< std::pair< int, double > >::value == 2, "");

        static_assert(std::is_same< rename< std::tuple, list< int, double > >, std::tuple< int, double > >{}, "");

        template < class T >
        using add_pointer_t = typename std::add_pointer< T >::type;

        static_assert(std::is_same< transform< add_pointer_t, list< int, void > >, list< int *, void * > >{}, "");
        static_assert(std::is_same< transform< add_pointer_t, list<> >, list<> >{}, "");

        static_assert(st_contains< int, list< int, bool > >{}, "");
        static_assert(!st_contains< double, list< int, bool > >{}, "");

        using map = list< list< int, void * >, list< void, double * >, list< float, double * > >;
        static_assert(std::is_same< mp_find< map, int >, list< int, void * > >{}, "");
        static_assert(std::is_same< mp_find< map, double >, void >{}, "");

        static_assert(std::is_same< repeat< 0, int >, list<> >{}, "");
        static_assert(std::is_same< repeat< 3, int >, list< int, int, int > >{}, "");

        static_assert(std::is_same< drop_front< 0, list< int, double > >, list< int, double > >{}, "");
        static_assert(std::is_same< drop_front< 1, list< int, double > >, list< double > >{}, "");
        static_assert(std::is_same< drop_front< 2, list< int, double > >, list<> >{}, "");

        static_assert(std::is_same< at< 0, list< int, double > >, int >{}, "");
        static_assert(std::is_same< at< 1, list< int, double > >, double >{}, "");

        static_assert(std::is_same< repeat_c< 2, int, 42 >, gt_integer_sequence< int, 42, 42 > >{}, "");

        static_assert(conjunction<>{}, "");
        static_assert(conjunction< std::true_type, std::true_type >{}, "");
        static_assert(!conjunction< std::true_type, std::false_type >{}, "");
        static_assert(!conjunction< std::false_type, std::true_type >{}, "");
        static_assert(!conjunction< std::false_type, std::false_type >{}, "");

        static_assert(!disjunction<>{}, "");
        static_assert(disjunction< std::true_type, std::true_type >{}, "");
        static_assert(disjunction< std::true_type, std::false_type >{}, "");
        static_assert(disjunction< std::false_type, std::true_type >{}, "");
        static_assert(!disjunction< std::false_type, std::false_type >{}, "");

        static_assert(std::is_same< st_make_index_map< list< int, void > >,
                          list< list< int, std::integral_constant< size_t, 0 > >,
                                        list< void, std::integral_constant< size_t, 1 > > > >{},
            "");

        static_assert(st_position< int, list< int, double > >{} == 0, "");
        static_assert(st_position< int, list< double, int > >{} == 1, "");
        static_assert(st_position< void, list< double, int > >{} == 2, "");

        static_assert(
            std::is_same< st_positions< list< int, double >, list< int, double > >, gt_index_sequence< 0, 1 > >{}, "");
        static_assert(
            std::is_same< st_positions< list< double, int >, list< int, double > >, gt_index_sequence< 1, 0 > >{}, "");

        template < class... >
        struct f;
        static_assert(std::is_same< combine< f, repeat< 8, int > >,
                          f< f< f< int, int >, f< int, int > >, f< f< int, int >, f< int, int > > > >{},
            "");

        static_assert(std::is_same< concat< list< int > >, list< int > >{}, "");
        static_assert(std::is_same< concat< list< int >, list< void > >, list< int, void > >{}, "");
        static_assert(std::is_same< concat< list< int >, list< void, double >, list< void, int > >,
                          list< int, void, double, void, int > >{},
            "");

        static_assert(std::is_same< filter< std::is_pointer, list< void, int *, double, double ** > >,
                          list< int *, double ** > >{},
            "");

        static_assert(all_of< is_list, list< list<>, list< int > > >{}, "");

        static_assert(std::is_same< dedup< list<> >, list<> >{}, "");
        static_assert(std::is_same< dedup< list< int, void > >, list< int, void > >{}, "");
        static_assert(std::is_same< dedup< list< int, void, void, void, int, void > >, list< int, void > >{}, "");

        static_assert(std::is_same< zip< list< int >, list< void > >, list< list< int, void > > >{}, "");
        static_assert(
            std::is_same<
                zip< list< int, int *, int ** >, list< void, void *, void ** >, list< char, char *, char ** > >,
                list< list< int, void, char >, list< int *, void *, char * >, list< int **, void **, char ** > > >{},
            "");
    }
}

TEST(dummy, dummy) {}
