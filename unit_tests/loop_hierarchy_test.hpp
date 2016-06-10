/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include <stencil_composition/loop_hierarchy.hpp>
#define VERBOSE 1

namespace loop_test {
    using namespace gridtools;

    // stub iterate_domain
    struct iterate_domain_ {

        template < typename Index >
        void get_index(Index idx) const {}

        template < typename Index >
        void set_index(Index idx) {}

        template < ushort_t index, typename Step >
        void increment() {}
    };

    struct functor {

        functor() : m_iterations(0) {}

        void operator()() { m_iterations++; }

        uint_t m_iterations;
    };

    bool test() {

        typedef array< uint_t, 3 > array_t;
        iterate_domain_ it_domain;
        functor fun;

        loop_hierarchy< array_t,
            loop_item< 1, int_t, 1 >,
            loop_item< 5, short_t, 1 >
#ifdef CXX11_ENABLED
            ,
            static_loop_item< 0, 0u, 10u, uint_t, 1 >
#endif
            > h(2, 5, 6, 8);
        h.apply(it_domain, fun);

        return fun.m_iterations ==
               4 * 3
#ifdef CXX11_ENABLED
                   * 11
#endif
            ;
    }
} // namespace loop_test
