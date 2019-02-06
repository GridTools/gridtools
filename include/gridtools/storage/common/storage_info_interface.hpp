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

#include <type_traits>

#include "../../common/array.hpp"
#include "../../common/array_dot_product.hpp"
#include "../../common/defs.hpp"
#include "../../common/error.hpp"
#include "../../common/generic_metafunctions/accumulate.hpp"
#include "../../common/generic_metafunctions/binary_ops.hpp"
#include "../../common/generic_metafunctions/is_all_integrals.hpp"
#include "../../common/host_device.hpp"
#include "../../common/layout_map.hpp"
#include "../../meta/type_traits.hpp"
#include "../../meta/utility.hpp"
#include "alignment.hpp"
#include "halo.hpp"
#include "storage_info_metafunctions.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    namespace impl_ {

        /**
         * @brief Internal helper function to check if two given storage infos contain the same information.
         * The function performs checks on all dimensions. This function is the base case.
         * @return true if the dimension, stride, size, initial_offset, etc. is equal, otherwise false
         */
        template <uint_t N, typename StorageInfo>
        GT_FUNCTION enable_if_t<N == 0, bool> equality_check(StorageInfo a, StorageInfo b) {
            return (a.template total_length<N>() == b.template total_length<N>()) &&
                   (a.template stride<N>() == b.template stride<N>()) && (a.length() == b.length()) &&
                   (a.total_length() == b.total_length()) && (a.padded_total_length() == b.padded_total_length());
        }

        /**
         * @brief Internal helper function to check if two given storage infos contain the same information.
         * The function performs checks on all dimensions. This function is the step case.
         * @return true if the dimension, stride, size, initial_offset, etc. is equal, otherwise false
         */
        template <uint_t N, typename StorageInfo>
        GT_FUNCTION enable_if_t<N != 0, bool> equality_check(StorageInfo a, StorageInfo b) {
            return (a.template total_length<N>() == b.template total_length<N>()) &&
                   (a.template stride<N>() == b.template stride<N>()) && equality_check<N - 1>(a, b);
        }
    } // namespace impl_

    /**
     * @brief The storage info interface. This class contains methods that should be implemented by all storage_info
     * implementations.
     * @tparam Id unique ID that should be shared among all storage infos with the same dimensionality.
     * @tparam Layout information about the memory layout
     * @tparam Halo information about the halo sizes (by default no halo is set)
     * @tparam Alignment information about the alignment
     */
    template <uint_t Id,
        typename Layout,
        typename Halo = zero_halo<Layout::masked_length>,
        typename Alignment = alignment<1>>
    struct storage_info_interface;

    template <uint_t Id, int... LayoutArgs, uint_t... Halos, typename Align>
    struct storage_info_interface<Id, layout_map<LayoutArgs...>, halo<Halos...>, Align> {
        using layout_t = layout_map<LayoutArgs...>;
        using halo_t = halo<Halos...>;
        using alignment_t = Align;
        static const int max_layout_v = layout_t::max();

        GT_STATIC_ASSERT((sizeof...(Halos) == layout_t::masked_length),
            GT_INTERNAL_ERROR_MSG("Halo size does not match number of dimensions"));
        GT_STATIC_ASSERT(
            is_alignment<Align>::value, GT_INTERNAL_ERROR_MSG("Given type is not an alignment type"));

        static constexpr uint_t ndims = layout_t::masked_length;

      private:
        using this_t = storage_info_interface<Id, layout_map<LayoutArgs...>, halo<Halos...>, Align>;
        array<uint_t, ndims> m_total_lengths;
        array<uint_t, ndims> m_padded_lengths;
        array<uint_t, ndims> m_strides;

        /*
            When computing the size of a storage, either length,
            total_length, or padded_total_length, we need to multiply
            the dimensions for those dimensions that are not
            associated to a -1 (masked-dimension).

            In addition when computing the size of the inner region, we need to
            remove the halos from the sizes.
         */
        template <uint_t... Idxs, typename Array, typename Halo = zero_halo<ndims>>
        GT_FUNCTION static constexpr uint_t multiply_if_layout(
            meta::integer_sequence<uint_t, Idxs...>, Array const &array, Halo h = zero_halo<ndims>{}) {
            return accumulate(
                multiplies(), ((layout_t::template at<Idxs>() >= 0) ? array[Idxs] - 2 * h.at(Idxs) : 1)...);
        }

        template <uint_t... Seq, typename... Ints>
        GT_FUNCTION constexpr int offset(meta::integer_sequence<uint_t, Seq...>, Ints... idx) const {
            return accumulate(plus_functor(), (idx * m_strides[Seq])...);
        }

        template <int... Inds>
        GT_FUNCTION constexpr int first_index_impl(meta::integer_sequence<int, Inds...>) const {
            return index(halo_t::template at<Inds>()...);
        }

        template <uint_t... Ints, typename... Coords>
        GT_FUNCTION constexpr bool check_bounds(meta::integer_sequence<uint_t, Ints...>, Coords... coords) const {
            return accumulate(logical_and(),
                true,
                ((layout_t::template at<Ints>() < 0) or (((int)coords >= 0) and (coords < m_total_lengths[Ints])))...);
        }

      public:
        constexpr static uint_t id = Id;

        constexpr storage_info_interface() = delete;

        /**
         * @brief storage info constructor. Additionally to initializing the members the halo
         * region is added to the corresponding dimensions and the alignment is applied.
         */
        template <typename... Dims,
            enable_if_t<sizeof...(Dims) == ndims && is_all_integral_or_enum<Dims...>::value, int> = 0>
        GT_FUNCTION constexpr storage_info_interface(Dims... dims_)
            : m_total_lengths{static_cast<uint_t>(dims_)...},
              m_padded_lengths{pad_dimensions<alignment_t, max_layout_v, LayoutArgs>(
                  handle_masked_dims<LayoutArgs>::extend(dims_))...},
              m_strides(get_strides<layout_t>::get_stride_array(pad_dimensions<alignment_t, max_layout_v, LayoutArgs>(
                  handle_masked_dims<LayoutArgs>::extend(dims_))...)) {}

        GT_FUNCTION
        storage_info_interface(array<uint_t, ndims> dims, array<uint_t, ndims> strides)
            : m_total_lengths{dims}, m_strides(strides) {

            // We guess the padded lengths from the dimensions and the strides. Assume, that the strides are sorted,
            // e.g., [256, 16, 1], and the dimensions are [5, 9, 9]. For the largest stride, we assume that padding
            // = dimension (e.g. in this example the i-padding is 5). For all others we can calculate the padding from
            // the strides (e.g. in this example, the j-padding is 256 / 16 = 16, and the k-padding is 16 / 1 = 1).
            auto sorted_strides = strides;
            for (uint_t i = 0; i < ndims; ++i)
                for (uint_t j = i + 1; j < ndims; ++j)
                    if (sorted_strides[i] > sorted_strides[j]) {
                        auto tmp = sorted_strides[i];
                        sorted_strides[i] = sorted_strides[j];
                        sorted_strides[j] = tmp;
                    }

            for (uint_t i = 0; i < ndims; ++i) {
                if (strides[i] == sorted_strides[ndims - 1])
                    m_padded_lengths[i] = dims[i];
                else if (strides[i] == 0) {
                    m_padded_lengths[i] = 0;
                } else {
                    for (int j = i; j < ndims; ++j)
                        if (strides[i] != sorted_strides[j]) {
                            m_padded_lengths[i] = sorted_strides[j] / strides[i];
                            break;
                        }
                }
            }
        }

        /**
         * @brief storage info copy constructor.
         */
        constexpr storage_info_interface(storage_info_interface const &other) = default;

        /**
         * @brief member function to retrieve the total size (dimensions, halos, initial_offset, padding).
         * @return total size including dimensions, halos, initial_offset, padding, and initial_offset
         */
        GT_FUNCTION constexpr uint_t padded_total_length() const {
            return multiply_if_layout(meta::make_integer_sequence<uint_t, ndims>{}, m_padded_lengths);
        }

        /**
         * @brief member function to retrieve the number of domain elements
         * (dimensions, halos, no initial_offset, no padding).
         * @return number of domain elements
         */
        GT_FUNCTION constexpr uint_t total_length() const {
            return multiply_if_layout(meta::make_integer_sequence<uint_t, ndims>{}, m_total_lengths);
        }

        /**
         * @brief member function to retrieve the number of inner domain elements
         * (dimensions, no halos, no initial_offset, no padding).
         * @return number of inner domain elements
         */
        GT_FUNCTION constexpr uint_t length() const {
            return multiply_if_layout(meta::make_integer_sequence<uint_t, ndims>{}, m_total_lengths, halo_t{});
        }

        /**
         * @brief Returns the array of total_lengths, the lengths including the halo points (the outer region)
         */
        GT_FUNCTION constexpr const array<uint_t, ndims> &total_lengths() const { return m_total_lengths; }

        /**
         * @brief deprecated, see total_lengths()
         */
        GT_DEPRECATED("dims() is deprecated, use total_lengths() (deprecated after 1.07.00)")
        GT_FUNCTION constexpr const array<uint_t, ndims> &dims() const { return total_lengths(); }

        /*
         * @brief Returns the length of a dimension including the halo points (the outer region)
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr int total_length() const {
            GT_STATIC_ASSERT(
                (Dim < ndims), GT_INTERNAL_ERROR_MSG("Out of bounds access in storage info dimension call."));
            return m_total_lengths[Dim];
        }

        /**
         * @brief deprecated: see total_length()
         */
        template <uint_t Dim>
        GT_DEPRECATED("dim<Dim>() is deprecated, use total_length<Dim>() (deprecated after 1.07.00)")
        GT_FUNCTION constexpr uint_t dim() const {
            return total_length<Dim>();
        }

        /**
         * @brief Returns the length of a dimension including the halo points (the outer region) and padding.
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr uint_t padded_length() const {
            return m_padded_lengths[Dim];
        }

        /**
         * @brief Returns the array of padded_lengths, the lengths including the halo points (the outer region) and
         * padding.
         */
        GT_FUNCTION constexpr const array<uint_t, ndims> &padded_lengths() const { return m_padded_lengths; }

        /**
         * @brief Returns the length of a dimension excluding the halo points (only the inner region)
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr uint_t length() const {
            return m_total_lengths[Dim] - 2 * halo_t::template at<Dim>();
        }

        /**
         * @brief Returns the index of the first element in the specified dimension when iterating in the whole outer
         * region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr uint_t total_begin() const {
            return 0;
        }

        /**
         * @brief Returns the index of the last element in the specified dimension when iterating in the whole outer
         * region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr uint_t total_end() const {
            return total_length<Dim>() - 1;
        }

        /**
         * @brief Returns the index of the first element in the specified dimension when iterating in the inner region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr uint_t begin() const {
            return halo_t::template at<Dim>();
        }

        /**
         * @brief Returns the index of the last element in the specified dimension when iterating in the inner region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr uint_t end() const {
            return begin<Dim>() + length<Dim>() - 1;
        }

        /**
         * @brief member function to retrieve the (aligned) stride (e.g., I, J, or K)
         * @tparam Coord queried coordinate
         * @return aligned stride size
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr uint_t stride() const {
            GT_STATIC_ASSERT(
                (Dim < ndims), GT_INTERNAL_ERROR_MSG("Out of bounds access in storage info stride call."));
            return get<Dim>(m_strides);
        }

        /**
         * @brief return the array of (aligned) strides, see stride() for details.
         */
        GT_FUNCTION constexpr const array<uint_t, ndims> &strides() const { return m_strides; }

        /**
         * @brief member function to retrieve an offset (or index) when given offsets in I,J,K, etc.
         * E.g., index(1,2,3) --> 1*strideI + 2*strideJ + 3*strideK + initial_offset
         * @param idx given offsets
         * @return index
         */

        template <typename... Ints,
            enable_if_t<sizeof...(Ints) == ndims && is_all_integral_or_enum<Ints...>::value, int> = 0>
        GT_FUNCTION constexpr int index(Ints... idx) const {
#ifdef NDEBUG
            return offset(meta::make_integer_sequence<uint_t, ndims>{}, idx...);
#else
            return error_or_return(check_bounds(meta::make_integer_sequence<uint_t, ndims>{}, idx...),
                offset(meta::make_integer_sequence<uint_t, ndims>{}, idx...),
                "Storage out of bounds access");
#endif
        }

        /**
         * @brief member function to retrieve an offset (or index) when given an array of offsets in I,J,K, etc.
         * E.g., index(1,2,3) --> 1*strideI + 2*strideJ + 3*strideK + initial_offset
         * @param offsets given offset array
         * @return index
         */
        GT_FUNCTION constexpr int index(gridtools::array<int, ndims> const &offsets) const {
            return array_dot_product(offsets, m_strides);
        }

        GT_FUNCTION constexpr int first_index_of_inner_region() const {
            return first_index_impl(meta::make_integer_sequence<int, ndims>{});
        }

        /**
         * @brief function to check for equality of two given storage_infos
         * @param rhs right hand side storage info instance
         * @return true if the storage infos are equal, false otherwise
         */
        GT_FUNCTION bool operator==(this_t const &rhs) const { return impl_::equality_check<ndims - 1>(*this, rhs); }

        GT_FUNCTION bool operator!=(this_t const &rhs) const { return !operator==(rhs); }
    };

    template <typename T>
    struct is_storage_info : std::false_type {};

    template <uint_t Id, typename Layout, typename Halo, typename Alignment>
    struct is_storage_info<storage_info_interface<Id, Layout, Halo, Alignment>> : std::true_type {};

    /**
     * @}
     */
} // namespace gridtools
