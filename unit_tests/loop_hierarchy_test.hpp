/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <stencil-composition/loop_hierarchy.hpp>
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
