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
