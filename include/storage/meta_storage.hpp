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

#include "../common/gpu_clone.hpp"
#include "./meta_storage_base.hpp"
#include "./meta_storage_tmp.hpp"
#include "./meta_storage_aligned.hpp"
#ifdef CXX11_ENABLED
#include "../common/generic_metafunctions/repeat_template.hpp"
#include "../common/generic_metafunctions/variadic_to_vector.hpp"
#include <boost/type_traits/is_integral.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#endif
/**
   @file
   @brief implementation of a container for the storage meta information
*/

/**
   @class
   @brief containing the meta information which is clonable to GPU

   double inheritance is used in order to make the storage metadata clonable to the gpu.

   NOTE: Since this class is inheriting from clonable_to_gpu, the constexpr property of the
   meta_storage_base is lost (this can be easily avoided on the host)
*/
namespace gridtools {

    template < typename BaseStorage >
    struct meta_storage : public BaseStorage {

        static const bool is_temporary = BaseStorage::is_temporary;
        typedef BaseStorage super;
        typedef typename BaseStorage::basic_type basic_type;
        typedef typename BaseStorage::index_type index_type;
        typedef meta_storage< BaseStorage > original_storage;

        using super::space_dimensions;

        /** @brief copy ctor

            forwarding to the base class
        */
        GT_FUNCTION constexpr meta_storage(meta_storage< BaseStorage > const &other) : super(other) {}

#if defined(CXX11_ENABLED)
        /** @brief ctor

            forwarding to the base class
        */
        template < typename... IntTypes,
            typename Dummy = typename boost::enable_if_c<
                boost::is_integral< typename boost::mpl::at_c< typename variadic_to_vector< IntTypes... >::type,
                    0 >::type >::type::value,
                bool >::type >
        constexpr meta_storage(IntTypes... args)
            : super(args...) {}

        /**@brief operator equals (same dimension size, etc.) */
        constexpr bool operator==(const meta_storage &other) const { return super::operator==(other); }

        constexpr meta_storage(array< uint_t, space_dimensions > const &a) : super(a) {}
#else
        // constructor picked in absence of CXX11 or with GCC<4.9
        /** @brief ctor

            forwarding to the base class
        */
        explicit constexpr meta_storage(uint_t dim1, uint_t dim2, uint_t dim3) : super(dim1, dim2, dim3) {}

        /** @brief ctor

            forwarding to the base class
        */
        constexpr meta_storage(uint_t const &initial_offset_i,
            uint_t const &initial_offset_j,
            uint_t const &dim3,
            uint_t const &n_i_threads,
            uint_t const &n_j_threads)
            : super(initial_offset_i, initial_offset_j, dim3, n_i_threads, n_j_threads) {}

        /**@brief operator equals (same dimension size, etc.) */
        bool operator==(const meta_storage &other) const { return super::operator==(other); }

#endif

#ifndef STRUCTURED_GRIDS
        // API for icosahedral grid only
        /**@brief straightforward interface*/
        template < typename... T >
        constexpr GT_FUNCTION static int _index(T const &... args_) {
            return super::_index(args_...);
        }

        template < typename LayoutT, typename StridesVector >
        GT_FUNCTION static constexpr int_t _index(
            StridesVector const &RESTRICT strides_, array< int_t, space_dimensions > const &offsets) {
            GRIDTOOLS_STATIC_ASSERT((is_layout_map< LayoutT >::value), "wrong type");
            return super::template _index< LayoutT, StridesVector >(strides_, offsets);
        }
#endif // GRIDBACKEND==icosahedral

#ifndef __CUDACC__
      private:
#endif
        /** @brief empty ctor

            should never be called
            (only by nvcc because it does not compile the parallel_storage CXX11 version)
        */
        explicit meta_storage() : super() {}
    };

    /** \addtogroup specializations Specializations
        Partial specializations
        @{
    */

    template < typename T >
    struct is_meta_storage;

    template < typename Storage >
    struct is_meta_storage< meta_storage< Storage > > : boost::mpl::true_ {};

    template < typename Storage >
    struct is_meta_storage< meta_storage< Storage > & > : boost::mpl::true_ {};

    template < typename Storage >
    struct is_meta_storage< no_meta_storage_type_yet< Storage > > : is_meta_storage< Storage > {};

#ifdef CXX11_ENABLED
    template < typename Index, typename Layout, bool IsTemporary, typename... Whatever >
    struct is_meta_storage< meta_storage_base< Index, Layout, IsTemporary, Whatever... > > : boost::mpl::true_ {};
#else
    template < typename Index, typename Layout, bool IsTemporary, typename TileI, typename TileJ >
    struct is_meta_storage< meta_storage_base< Index, Layout, IsTemporary, TileI, TileJ > > : boost::mpl::true_ {};
#endif

    template < typename T >
    struct is_ptr_to_meta_storage : boost::mpl::false_ {};

    template < typename T >
    struct is_ptr_to_meta_storage< pointer< const T > > : is_meta_storage< T > {};

    /**@}*/

} // namespace gridtools
