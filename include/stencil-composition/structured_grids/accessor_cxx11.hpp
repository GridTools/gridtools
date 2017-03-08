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
#include "../accessor_base.hpp"
#include "../arg.hpp"
#include "../../common/dimension.hpp"
#include "../../common/generic_metafunctions/all_integrals.hpp"
#include "../../common/generic_metafunctions/static_if.hpp"

/**
   @file

   @brief File containing the definition of the placeholders used to
   address the storage from whithin the functors.  A placeholder is an
   implementation of the proxy design pattern for the storage class,
   i.e. it is a light object used in place of the storage when
   defining the high level computations, and it will be bound later on
   with a specific instantiation of a storage class.

   Two different types of placeholders are considered:

   - arg represents the storage in the body of the main function, and
     it gets lazily assigned to a real storage.

   - accessor represents the storage inside the functor struct
     containing a Do method. It can be instantiated directly in the Do
     method, or it might be a constant expression instantiated outside
     the functor scope and with static duration.
*/

namespace gridtools {

    /**
       @brief the definition of accessor visible to the user

       \tparam ID the integer unic ID of the field placeholder

       \tparam Extent the extent of i/j indices spanned by the
               placeholder, in the form of <i_minus, i_plus, j_minus,
               j_plus>.  The values are relative to the current
               position. See e.g. horizontal_diffusion::out_function
               as a usage example.

       \tparam Number the number of dimensions accessed by the
               field. Notice that we don't distinguish at this level what we
               call "space dimensions" from the "field dimensions". Both of
               them are accessed using the same interface. whether they are
               field dimensions or space dimension will be decided at the
               moment of the storage instantiation (in the main function)
     */
    template < uint_t ID,
        enumtype::intend Intend = enumtype::in,
        typename Extent = extent< 0, 0, 0, 0, 0, 0 >,
        ushort_t Number = 3 >
    struct accessor : public accessor_base< ID, Intend, Extent, Number > {
        typedef accessor_base< ID, Intend, Extent, Number > super;
        typedef typename super::index_type index_type;
        typedef typename super::offset_tuple_t offset_tuple_t;

        GT_FUNCTION
        constexpr accessor() : super() {}

        /**inheriting all constructors from accessor_base*/
        using super::accessor_base;

        // workaround in order to support using the "normal" accessor even for non rectangular data fields
        // see the implementation of get_data_field_index (specialization of non rectangular data field):
        // for a non rectangular data field, the components need to be accessed with the constexpr (in order
        // to avoid having to loop over all components in order to compute the offset of certain snapshot).
        // That forces to use an accessor_mixed, unless we are only accessing the first component.
        // In order to allow accessing the first component with a "normal" accessor we need to provide this
        // dummy get_constexpr that returns always 0
        template < short_t Idx >
        GT_FUNCTION static constexpr int_t get_constexpr() {
            return 0;
        }
    };

    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using in_accessor = accessor< ID, enumtype::in, Extent, Number >;

    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using inout_accessor = accessor< ID, enumtype::inout, Extent, Number >;

} // namespace gridtools
