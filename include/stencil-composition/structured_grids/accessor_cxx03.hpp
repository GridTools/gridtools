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
        typedef typename super::index_t index_t;
        typedef typename super::offset_tuple_t offset_tuple_t;

#ifdef CXX11_ENABLED

        GT_FUNCTION
        constexpr accessor() : super() {}

        /**inheriting all constructors from offset_tuple*/
        using super::accessor_base;

#else

        // copy ctor
        GT_FUNCTION
        constexpr explicit accessor(accessor< ID, Intend, Extent, Number > const &other) : super(other) {}

        // copy ctor from an accessor with different ID
        template < ushort_t OtherID >
        GT_FUNCTION constexpr explicit accessor(const accessor< OtherID, Intend, Extent, Number > &other)
            : super(static_cast< accessor_base< OtherID, Intend, Extent, Number > >(other)) {}

        GT_FUNCTION
        constexpr explicit accessor() : super() {}

        /** @brief constructor forwarding all the arguments*/
        template < typename X, typename Y, typename Z, typename T >
        GT_FUNCTION constexpr accessor(X x, Y y, Z z, T t) : super(x, y, z, t) {}

        /** @brief constructor forwarding all the arguments*/
        template < typename X, typename Y, typename Z >
        GT_FUNCTION constexpr accessor(X x, Y y, Z z) : super(x, y, z) {}

        /** @brief constructor forwarding all the arguments*/
        template < typename X >
        GT_FUNCTION constexpr accessor(X x) : super(x) {}

        /** @brief constructor forwarding all the arguments*/
        template < typename X, typename Y >
        GT_FUNCTION constexpr accessor(X x, Y y) : super(x, y) {}

#endif
    };

#ifdef CXX11_ENABLED
    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using in_accessor = accessor< ID, enumtype::in, Extent, Number >;

    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using inout_accessor = accessor< ID, enumtype::inout, Extent, Number >;
#endif
} // namespace gridtools
