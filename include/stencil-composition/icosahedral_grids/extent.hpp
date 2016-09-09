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

namespace gridtools {

    template < int_t R = 0 >
    struct extent {
        static const int_t value = R;
    };

    template < typename T >
    struct is_extent : boost::mpl::false_ {};

    template < int R >
    struct is_extent< extent< R > > : boost::mpl::true_ {};

    /**
     * Metafunction taking two extents and yielding a extent containing them
     */
    template < typename Extent1, typename Extent2 >
    struct enclosing_extent {
        BOOST_MPL_ASSERT((is_extent< Extent1 >));
        BOOST_MPL_ASSERT((is_extent< Extent2 >));

        typedef extent< boost::mpl::max< static_uint< Extent1::value >, static_uint< Extent2::value > >::type::value >
            type;
    };

    /**
     * Metafunction to add two extents
     */
    template < typename Extent1, typename Extent2 >
    struct sum_extent {
        BOOST_MPL_ASSERT((is_extent< Extent1 >));
        BOOST_MPL_ASSERT((is_extent< Extent2 >));

        typedef extent< Extent1::value + Extent2::value > type;
    };

} // namespace gridtools
