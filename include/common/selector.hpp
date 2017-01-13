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
#include "generic_metafunctions/variadic_typedef.hpp"
#include "gt_math.hpp"
#include "generic_metafunctions/is_variadic_pack_of.hpp"

namespace gridtools {

#ifdef CXX11_ENABLED
    namespace impl {
        constexpr ushort_t compute_existing_dim() { return 0; }

        template < typename... Int >
        constexpr ushort_t compute_existing_dim(int d, Int... ds) {
            return d == 1 ? 1 + compute_existing_dim(ds...) : compute_existing_dim(ds...);
        }
        template < int_t Val >
        struct is_one {
            static constexpr bool value = ((Val == 1) || (Val == -1));
        };
    }

    /*
     * metafunction use to selects certain positions in a variadic pack
     * The variadic templates accept only 1 (select) or -1 (unselect)
     */
    template < int_t... Int >
    struct selector {
        static_assert((is_variadic_pack_of(impl::is_one< Int >::value...)), "ERROR");
        typedef variadic_typedef_c< int_t, Int... > indices;
        static constexpr ushort_t length = indices::length;

        static constexpr ushort_t existingdim_length = impl::compute_existing_dim(Int...);
        template < ushort_t Idx >
        struct get_elem {
            GRIDTOOLS_STATIC_ASSERT((Idx <= sizeof...(Int)), "Out of bound access in variadic pack");
            typedef typename indices::template get_elem< Idx >::type type;
            static constexpr const int_t value = type::value;
        };
    };

    template < typename T >
    struct is_selector : boost::mpl::false_ {};

    template < int_t... Int >
    struct is_selector< selector< Int... > > : boost::mpl::true_ {};
#endif

} // gridtools
