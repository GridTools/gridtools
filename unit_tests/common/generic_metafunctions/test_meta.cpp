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
        static_assert(has_type< lazy< void > >{}, "");
        static_assert(has_type< std::is_void< int > >{}, "");

        // is_meta_class
        static_assert(!is_meta_class< int >{}, "");
        static_assert(!is_meta_class< f< int > >{}, "");
        static_assert(!is_meta_class< std::is_void< int > >{}, "");
        static_assert(is_meta_class< always< int > >{}, "");
        static_assert(is_meta_class< quote< f > >{}, "");
        static_assert(is_meta_class< ctor< f<> > >{}, "");

        // length
        static_assert(length< list<> >::value == 0, "");
        static_assert(length< std::tuple< int > >::value == 1, "");
        static_assert(length< std::pair< int, double > >::value == 2, "");
        static_assert(length< f< int, int, double > >::value == 3, "");

        // compose
        static_assert(std::is_same< compose< f, g >::apply< int, void >, f< g< int, void > > >{}, "");

        // ctor
        static_assert(std::is_same< apply< ctor< f< double > >, int, void >, f< int, void > >{}, "");

        // rename
        static_assert(std::is_same< apply< rename< f >, list< int, double > >, f< int, double > >{}, "");

        // transform
        static_assert(std::is_same< apply< transform< f >, list<> >, list<> >{}, "");
        static_assert(std::is_same< apply< transform< f >, list< int, void > >, list< f< int >, f< void > > >{}, "");
        static_assert(
            std::is_same< apply< transform< f >, list< int, void >, list< int *, void * >, list< int **, void ** > >,
                list< f< int, int *, int ** >, f< void, void *, void ** > > >{},
            "");

        // st_contains
        static_assert(st_contains< list< int, bool >, int >{}, "");
        static_assert(!st_contains< list< int, bool >, void >{}, "");

        // mp_find
        using map = list< list< int, void * >, list< void, double * >, list< float, double * > >;
        static_assert(std::is_same< mp_find< map, int >, list< int, void * > >{}, "");
        static_assert(std::is_same< mp_find< map, double >, void >{}, "");

        // repeat
        static_assert(std::is_same< repeat< 0, int >, list<> >{}, "");
        static_assert(std::is_same< repeat< 3, int >, list< int, int, int > >{}, "");

        // drop_front
        static_assert(std::is_same< drop_front< 0, list< int, double > >, list< int, double > >{}, "");
        static_assert(std::is_same< drop_front< 1, list< int, double > >, list< double > >{}, "");
        static_assert(std::is_same< drop_front< 2, list< int, double > >, list<> >{}, "");

        // at
        static_assert(std::is_same< at_c< list< int, double >, 0 >, int >{}, "");
        static_assert(std::is_same< at_c< list< int, double >, 1 >, double >{}, "");

        // conjunction
        static_assert(conjunction<>{}, "");
        static_assert(conjunction< std::true_type, std::true_type >{}, "");
        static_assert(!conjunction< std::true_type, std::false_type >{}, "");
        static_assert(!conjunction< std::false_type, std::true_type >{}, "");
        static_assert(!conjunction< std::false_type, std::false_type >{}, "");

        // disjunction
        static_assert(!disjunction<>{}, "");
        static_assert(disjunction< std::true_type, std::true_type >{}, "");
        static_assert(disjunction< std::true_type, std::false_type >{}, "");
        static_assert(disjunction< std::false_type, std::true_type >{}, "");
        static_assert(!disjunction< std::false_type, std::false_type >{}, "");

        // st_position
        static_assert(st_position< list< int, double >, int >{} == 0, "");
        static_assert(st_position< list< double, int >, int >{} == 1, "");
        static_assert(st_position< list< double, int >, void >{} == 2, "");

        // combine
        static_assert(std::is_same< apply< combine< f >, list< int > >, int >{}, "");
        static_assert(std::is_same< apply< combine< f >, repeat< 8, int > >,
                          f< f< f< int, int >, f< int, int > >, f< f< int, int >, f< int, int > > > >{},
            "");

        // concat
        static_assert(std::is_same< concat< list< int > >, list< int > >{}, "");
        static_assert(std::is_same< concat< list< int >, list< void > >, list< int, void > >{}, "");
        static_assert(std::is_same< concat< list< int >, list< void, double >, list< void, int > >,
                          list< int, void, double, void, int > >{},
            "");

        // filter
        static_assert(std::is_same< apply< filter< std::is_pointer >, list< void, int *, double, double ** > >,
                          list< int *, double ** > >{},
            "");

        // all_of
        static_assert(all_of< is_list, list< list<>, list< int > > >{}, "");

        // dedup
        static_assert(std::is_same< dedup< list<> >, list<> >{}, "");
        static_assert(std::is_same< dedup< list< int, void > >, list< int, void > >{}, "");
        static_assert(std::is_same< dedup< list< int, void, void, void, int, void > >, list< int, void > >{}, "");

        // zip
        static_assert(std::is_same< zip< list< int >, list< void > >, list< list< int, void > > >{}, "");
        static_assert(
            std::is_same<
                zip< list< int, int *, int ** >, list< void, void *, void ** >, list< char, char *, char ** > >,
                list< list< int, void, char >, list< int *, void *, char * >, list< int **, void **, char ** > > >{},
            "");

        // bind
        static_assert(std::is_same< bind< f, _2, void, _1 >::apply< int, double >, f< double, void, int > >{}, "");

        // is_instantiation_of
        static_assert(is_instantiation_of< f >::apply< f<> >{}, "");
        static_assert(is_instantiation_of< f >::apply< f< int, void > >{}, "");
        static_assert(!is_instantiation_of< f >::apply< g<> >{}, "");
        static_assert(!is_instantiation_of< f >::apply< int >{}, "");
    }
}

TEST(dummy, dummy) {}
