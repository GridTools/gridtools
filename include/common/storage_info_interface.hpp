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

#include <utility>
#include <array>

#include <boost/type_traits.hpp>

#include "layout_map.hpp"
#include "halo.hpp"
#include "nano_array.hpp"
#include "alignment.hpp"
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

        const static int id = Id;

        template < typename... Dims >
        constexpr storage_info_interface(Dims... dims_)
            : m_dims{align_dimensions< Alignment, sizeof...(LayoutArgs), LayoutArgs >(
                  extend_by_halo< Halos, LayoutArgs >::extend(dims_))...},
              m_strides(get_strides< Layout >::get_stride_array(
                  align_dimensions< Alignment, sizeof...(LayoutArgs), LayoutArgs >(
                      extend_by_halo< Halos, LayoutArgs >::extend(dims_))...)),
              m_alignment(get_initial_offset< Layout, Alignment, Halo >::compute(),
                  nano_array< unsigned, sizeof...(Dims) >{
                      (unsigned)extend_by_halo< Halos, LayoutArgs >::extend(dims_)...},
                  get_strides< Layout >::get_stride_array(dims_...)) {
            static_assert((sizeof...(Dims) == Layout::length), "error");
        }

        constexpr storage_info_interface(storage_info_interface const &other) = default;

        constexpr unsigned size_part(unsigned start = Layout::length - 1) const {
            return (start == 0) ? m_dims[0] : m_dims[start] * size_part(start - 1);
        }

        constexpr unsigned size() const { return size_part() + m_alignment.m_initial_offset; }

        template < int Coord >
        constexpr int dim() const {
            return m_dims[Coord];
        }

        template < int Coord >
        constexpr int stride() const {
            return m_strides[Coord];
        }

        template < int Coord >
        constexpr int unaligned_dim() const {
            return m_alignment.template unaligned_dim< Coord >();
        }

        template < int Coord >
        constexpr int unaligned_stride() const {
            return m_alignment.template unaligned_stride< Coord >();
        }

        template < typename... Ints >
        constexpr int index_part(int cnt, int first, Ints... ints) const {
            return (cnt < Layout::length) ? first * m_strides[cnt] + index_part(cnt + 1, ints..., first) : 0;
        }

        template < typename... Ints >
        constexpr int index(Ints... idx) const {
            return index_part(0, idx...) + m_alignment.get_initial_offset();
        }

      private:
        nano_array< unsigned, Layout::length > m_dims;
        nano_array< unsigned, Layout::length > m_strides;
        alignment_impl< Alignment, Layout::length > m_alignment;
        constexpr storage_info_interface() {}
    };

    template < typename T >
    struct is_storage_info
        : boost::is_base_of<
              storage_info_interface< T::id, typename T::Layout, typename T::Halo, typename T::Alignment >,
              T > {};
}
