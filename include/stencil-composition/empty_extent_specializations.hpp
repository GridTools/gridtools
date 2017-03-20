/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

namespace gridtools {
    template < typename Extent1 >
    struct sum_extent< Extent1, empty_extent > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent1 >::value), "Type should be an Extent");
        typedef typename sum_extent< Extent1, extent<> >::type type;
    };

    template < typename Extent2 >
    struct sum_extent< empty_extent, Extent2 > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent2 >::value), "Type should be an Extent");
        typedef typename sum_extent< extent<>, Extent2 >::type type;
    };

    template <>
    struct sum_extent< empty_extent, empty_extent > {
        typedef typename sum_extent< extent<>, extent<> >::type type;
    };

    /**
     * Metafunction to check if a type is a extent - Specialization yielding true
     */
    template <>
    struct is_extent< empty_extent > : boost::true_type {};

    template < typename Extent1 >
    struct enclosing_extent< Extent1, empty_extent > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent1 >::value), "Type should be an Extent");
        typedef typename enclosing_extent< Extent1, extent<> >::type type;
    };

    template < typename Extent2 >
    struct enclosing_extent< empty_extent, Extent2 > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent2 >::value), "Type should be an Extent");
        typedef typename enclosing_extent< extent<>, Extent2 >::type type;
    };

    template <>
    struct enclosing_extent< empty_extent, empty_extent > {
        typedef typename enclosing_extent< extent<>, extent<> >::type type;
    };

} // namespace gridtools
