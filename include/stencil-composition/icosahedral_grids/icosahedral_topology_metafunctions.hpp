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

#pragma once
#include "../../common/defs.hpp"
#include "../../common/gt_math.hpp"
#include "../../common/selector.hpp"
#include "../../common/array.hpp"
#include "../location_type.hpp"
#include "../../common/generic_metafunctions/pack_get_elem.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"

namespace gridtools {
    namespace impl {

        /**
         * @brief Computes a unique identifier (to be used for metastorages) given a list of index values
         */
        template < uint_t Pos >
        GT_FUNCTION constexpr long long compute_uuid_selector(int cnt) {
            return 0;
        }

        /**
         * @brief Computes a unique identifier (to be used for metastorages) given a list of index values
         */
        template < uint_t Pos, typename... Int >
        GT_FUNCTION constexpr long long compute_uuid_selector(int cnt, int val0, Int... val) {
            return (cnt == 4) ? 0 : ((val0 == 1)
                                            ? gt_pow< Pos >::apply((long long)2) +
                                                  compute_uuid_selector< Pos + 1 >(cnt + 1, val...)
                                            : compute_uuid_selector< Pos + 1 >(cnt + 1, val...));
        }

        /**
         * Computes a unique identifier (to be used for metastorages) given the location type and a dim selector
         * that determines the dimensions of the layout map which are activated.
         * Only the first 4 dimension of the selector are considered, since the iteration space of the backend
         * does not make use of indices beyond the space dimensions
         */
        template < int_t LocationTypeIndex, typename Selector >
        struct compute_uuid {};

        template < int_t LocationTypeIndex, int_t... Int >
        struct compute_uuid< LocationTypeIndex, selector< Int... > > {
            static constexpr ushort_t value =
                enumtype::metastorage_library_indices_limit + LocationTypeIndex + compute_uuid_selector< 2 >(0, Int...);
        };

        /**
         * helper that initializes arrays of dimensions given an array of space dimensions and the rest of
         * extra dimensions of a storage
         */
        template < typename UInt, typename LocationType >
        struct array_elem_initializer {
            GRIDTOOLS_STATIC_ASSERT((is_location_type< LocationType >::value), "Error: expected a location type");

            template < int Idx >
            struct init_elem {
                GT_FUNCTION
                constexpr init_elem() {}

                GT_FUNCTION constexpr static UInt apply(const array< uint_t, 3 > space_dims) {
                    GRIDTOOLS_STATIC_ASSERT((Idx < 4), GT_INTERNAL_ERROR);
                    // cast to size_t to suppress a warning
                    return ((Idx == 0) ? space_dims[0]
                                       : ((Idx == 1) ? LocationType::n_colors::value : space_dims[(size_t)(Idx - 1)]));
                }

                template < typename... ExtraInts >
                GT_FUNCTION constexpr static UInt apply(const array< uint_t, 3 > space_dims, ExtraInts... extra_dims) {
                    // cast to size_t to suppress a warning
                    return ((Idx == 0) ? space_dims[0]
                                       : ((Idx == 1) ? LocationType::n_colors::value
                                                     : (Idx < 4 ? space_dims[(size_t)(Idx - 1)]
                                                                : pack_get_elem< Idx - 4 >::apply(extra_dims...))));
                }
            };
        };

        /**
         * @brief constructs an array containing the sizes of each dimension for a generic storage with any
         * number of dimensions.
         * It is formed from a basic array (with only the 3 space dimension sizes) and the specification
         * of the sizes of the extra dimensions passed as a variadic pack arguments. Example of use:
         *    array< uint_t, 6 > metastorage_sizes =
                impl::array_dim_initializers< uint_t, 6, cells >::apply(array<uint_t, 3>{1,3,5}, 7,9);
              will construct the array {1,2,3,5,7,9}
              (the size of the color dimension is added from the location type (cells) specified

         */
        template < typename Uint, size_t ArraySize, typename LocationType, typename Selector >
        struct array_dim_initializers;

        template < typename UInt, size_t ArraySize, typename LocationType, int_t... Ints >
        struct array_dim_initializers< UInt, ArraySize, LocationType, selector< Ints... > > {
            GRIDTOOLS_STATIC_ASSERT((is_location_type< LocationType >::value), "Error: expected a location type");

            template < typename... ExtraInts >
            GT_FUNCTION static constexpr array< UInt, ArraySize > apply(
                const array< uint_t, 3 > space_dims, ExtraInts... extra_dims) {
                using seq = apply_gt_integer_sequence< typename make_gt_integer_sequence< int, ArraySize >::type >;

                return seq::template apply< array< UInt, ArraySize >,
                    array_elem_initializer< UInt, LocationType >::template init_elem >(space_dims, extra_dims...);
            }
        };
    }
}
