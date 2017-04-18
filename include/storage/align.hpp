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
namespace gridtools {

    /**
       @brief returns the padding to be added at the end of a specific dimension

       used internally in the library to allocate the storages. By default only the stride 1 dimension is padded.
       The stride 1 dimension is identified at compile-time given the layout-map.
     */
    template < ushort_t Alignment, typename LayoutMap >
    struct align {

        // the stride is one when the value in the layout vector is the highest
        template < uint_t Coordinate >
        struct has_stride_one {
            static const bool value =
                (LayoutMap::template at_< Coordinate >::value == vec_max< typename LayoutMap::layout_vector_t >::value);
            typedef typename boost::mpl::bool_< value >::type type;
        };

        //     NOTE: nvcc does not understand that the functor below can be a constant expression
        /** applies the alignment to run-time values*/
        template < uint_t Coordinate, uint_t Halo, uint_t Padding >
        struct do_align {

            GT_FUNCTION
            static constexpr uint_t apply(uint_t const &dimension) {

                typedef static_uint< Halo + Padding > offset;

                // the stride is one when the value in the layout vector is the highest
                return (Alignment && ((dimension + offset::value) % Alignment) && has_stride_one< Coordinate >::value)
                           ? dimension + offset::value + Alignment - ((dimension + offset::value) % Alignment)
                           : dimension + offset::value;
            }
        };
    };

    /**@brief apply alignment to all coordinates regardless of the layout_map*/
    template < ushort_t Alignment, ushort_t Dimension >
    struct align_all {
        static const uint_t value =
            Alignment
                ? (Alignment && (Dimension % Alignment)) ? (Dimension + Alignment - (Dimension % Alignment)) : Dimension
                : Dimension;
    };

    /** @brief wrapper around the alignment boundary

        This class defines a keyword to be used when defining the storage
     */
    template < ushort_t Boundary >
    struct aligned {
        static const ushort_t value = Boundary;
    };

    template < typename T >
    struct is_aligned : boost::mpl::false_ {};

    template < ushort_t T >
    struct is_aligned< aligned< T > > : boost::mpl::true_ {};

} // namespace gridtools
