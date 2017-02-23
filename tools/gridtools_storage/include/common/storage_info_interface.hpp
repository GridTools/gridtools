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

#include <array>
#include <utility>

#include <boost/type_traits.hpp>

#include "alignment.hpp"
#include "defs.hpp"
#include "halo.hpp"
#include "layout_map.hpp"
#include "nano_array.hpp"
#include "storage_info_metafunctions.hpp"

namespace gridtools {

    template < unsigned Id,
        typename Layout,
        typename Halo = typename zero_halo< Layout::length >::type,
        typename Alignment = alignment< 0 > >
    struct storage_info_interface;

    template < unsigned Id, int... LayoutArgs, unsigned... Halos, typename Align >
    struct storage_info_interface< Id, layout_map< LayoutArgs... >, halo< Halos... >, Align > {
        using Layout = layout_map< LayoutArgs... >;
        using Halo = halo< Halos... >;
        using Alignment = Align;
        static_assert(sizeof...(Halos) == Layout::length, "Halo size does not match number of dimensions");
        static_assert(is_alignment<Align>::value, "Given type is not an alignment type");

        const static int id = Id;

        template < typename... Dims >
        constexpr storage_info_interface(Dims... dims_)
            : m_dims{align_dimensions< Alignment, sizeof...(LayoutArgs), LayoutArgs >(
                  extend_by_halo< Halos, LayoutArgs >::extend(dims_))...},
              m_strides(get_strides< Layout >::get_stride_array(
                  align_dimensions< Alignment, sizeof...(LayoutArgs), LayoutArgs >(
                      extend_by_halo< Halos, LayoutArgs >::extend(dims_))...)),
              m_alignment(nano_array< unsigned, sizeof...(Dims) >{(unsigned)extend_by_halo< Halos, LayoutArgs >::extend(
                              dims_)...},
                  get_strides< Layout >::get_stride_array(extend_by_halo< Halos, LayoutArgs >::extend(dims_)...)) {
            static_assert((sizeof...(Dims) == Layout::length), "Number of passed dimensions do not match the layout map length.");
        }

        constexpr storage_info_interface(storage_info_interface const &other) = default;

        GT_FUNCTION constexpr unsigned size() const { return size_part() + get_initial_offset(); }

        template < int Coord >
        GT_FUNCTION constexpr int dim() const {
            static_assert((Coord<Layout::length), "Out of bounds access in storage info dimension call.");
            return m_dims[Coord];
        }

        template < int Coord >
        GT_FUNCTION constexpr int stride() const {
            static_assert((Coord<Layout::length), "Out of bounds access in storage info stride call.");
            return m_strides[Coord];
        }

        template < int Coord >
        GT_FUNCTION constexpr int unaligned_dim() const {
            static_assert((Coord<Layout::length), "Out of bounds access in storage info unaligned dimension call.");
            return m_alignment.template unaligned_dim< Coord >() ? m_alignment.template unaligned_dim< Coord >()
                                                                 : dim< Coord >();
        }

        template < int Coord >
        GT_FUNCTION constexpr int unaligned_stride() const {
            static_assert((Coord<Layout::length), "Out of bounds access in storage info unaligned stride call.");
            return m_alignment.template unaligned_stride< Coord >() ? m_alignment.template unaligned_stride< Coord >()
                                                                    : stride< Coord >();
        }

        template < typename... Ints >
        GT_FUNCTION constexpr int index(Ints... idx) const {
            static_assert(sizeof...(Ints) == Layout::length, "Index function called with wrong number of arguments.");
            #ifdef NDEBUG
            return index_part< 0 >(idx...) + get_initial_offset();
            #else
            return (check_bounds<0>(idx...)) ? index_part< 0 >(idx...) + get_initial_offset() : error::trigger("Storage out of bounds access");
            #endif
        }

        GT_FUNCTION nano_array<unsigned, Layout::length> const& strides() const {
            return m_strides;
        }

        GT_FUNCTION static constexpr unsigned get_initial_offset() {
            return alignment_impl< Alignment, Layout, Halo >::InitialOffset;
        }

      private:
        nano_array< unsigned, Layout::length > m_dims;
        nano_array< unsigned, Layout::length > m_strides;
        alignment_impl< Alignment, Layout, Halo > m_alignment;
        constexpr storage_info_interface() {}

        template < unsigned N, typename... Ints >
        GT_FUNCTION constexpr typename boost::enable_if_c<(N<=Layout::length-1), bool>::type
        check_bounds(Ints... idx) const {
            // check for out of bounds access; each index is checked if it does not exceed the unaligned dimension.
            // masked dimensions are skipped and a recursive call is performed until all indices are checked.
            return ((Layout::template at<N>() == -1) || (get_value_from_pack(N, idx...) < unaligned_dim<N>())) && check_bounds<N+1>(idx...);
        }

        template < unsigned N, typename... Ints >
        GT_FUNCTION constexpr typename boost::enable_if_c<(N==Layout::length), bool>::type
        check_bounds(Ints... idx) const {
            // base case out of bounds check
            return true;
        }

        template < unsigned N, typename... Ints >
        GT_FUNCTION constexpr typename boost::enable_if_c< (N < Layout::length), int >::type index_part(
            int first, Ints... ints) const {
            return first * m_strides[N] + index_part< N + 1 >(ints..., first);
        }

        template < unsigned N, typename... Ints >
        GT_FUNCTION constexpr typename boost::enable_if_c< (N == Layout::length), int >::type index_part(
            int first, Ints... ints) const {
            return 0;
        }

        template < unsigned From = Layout::length - 1 >
        GT_FUNCTION constexpr typename boost::enable_if_c< (From > 0), unsigned >::type size_part() const {
            return m_dims[From] * size_part< From - 1 >();
        }

        template < unsigned From = Layout::length - 1 >
        GT_FUNCTION constexpr typename boost::enable_if_c< (From == 0), unsigned >::type size_part() const {
            return m_dims[0];
        }
    };

    template < typename T >
    struct is_storage_info
        : boost::is_base_of<
              storage_info_interface< T::id, typename T::Layout, typename T::Halo, typename T::Alignment >,
              T > {};
}
