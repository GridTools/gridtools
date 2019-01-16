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

#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/max_element.hpp>
#include <boost/mpl/modulus.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/size.hpp>
#include <boost/utility/enable_if.hpp>

#include "../common/array.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../common/generic_metafunctions/static_if.hpp"
#include "../meta/macros.hpp"
#include "../meta/make_indices.hpp"
#include "../meta/st_contains.hpp"
#include "../meta/st_position.hpp"
#include "../meta/type_traits.hpp"
#include "arg.hpp"
#include "block.hpp"
#include "expressions/expressions.hpp"
#include "offset_computation.hpp"
#include "pos3.hpp"
#include "run_functor_arguments.hpp"
#include "tmp_storage.hpp"

/**
   @file
   @brief file implementing helper functions which are used in iterate_domain to assign/increment strides, access
   indices and storage pointers.

   All the helper functions use template recursion to implement loop unrolling
*/

namespace gridtools {

    /**
       @brief struct to allocate recursively all the strides with the proper dimension

       the purpose of this struct is to allocate the storage for the strides of a set of storages. Tipically
       it is used to cache these strides in a fast memory (e.g. shared memory).
       \tparam ID recursion index, representing the current storage
       \tparam StorageInfoList typelist of the storages
    */
    template <ushort_t ID, typename StorageInfoList>
    struct strides_cached /** @cond */ : public strides_cached<ID - 1, StorageInfoList> /** @endcond */ {
        GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<StorageInfoList>::value > ID,
            GT_INTERNAL_ERROR_MSG("strides index exceeds the number of storages"));
        typedef typename boost::mpl::at_c<StorageInfoList, ID>::type storage_info_ptr_t;
        typedef
            typename boost::remove_pointer<typename boost::remove_cv<storage_info_ptr_t>::type>::type storage_info_t;
        typedef strides_cached<ID - 1, StorageInfoList> super;
        typedef array<int_t, storage_info_t::layout_t::masked_length - 1> data_array_t;

        template <short_t Idx>
        using return_t = typename boost::mpl::
            if_<boost::mpl::bool_<Idx == ID>, data_array_t, typename super::template return_t<Idx>>::type;

        /**@brief constructor, doing nothing more than allocating the space*/
        GT_FUNCTION
        strides_cached() : super() {}

        template <short_t Idx>
        GT_FUNCTION return_t<Idx> const &RESTRICT get() const {
            return static_if<(Idx == ID)>::apply(m_data, super::template get<Idx>());
        }

        template <short_t Idx>
        GT_FUNCTION return_t<Idx> &RESTRICT get() {
            return static_if<(Idx == ID)>::apply(m_data, super::template get<Idx>());
        }

      private:
        data_array_t m_data;
        strides_cached(strides_cached const &);
    };

    /**specialization to stop the recursion*/
    template <typename StorageInfoList>
    struct strides_cached<(ushort_t)0, StorageInfoList> {
        typedef typename boost::mpl::at_c<StorageInfoList, 0>::type storage_info_ptr_t;
        typedef
            typename boost::remove_pointer<typename boost::remove_cv<storage_info_ptr_t>::type>::type storage_info_t;

        GT_FUNCTION
        strides_cached() {}

        typedef array<int_t, storage_info_t::layout_t::masked_length - 1> data_array_t;

        template <short_t Idx>
        using return_t = data_array_t;

        template <short_t Idx>
        GT_FUNCTION data_array_t &RESTRICT get() { // stop recursion
            return m_data;
        }

        template <short_t Idx>
        GT_FUNCTION data_array_t const &RESTRICT get() const { // stop recursion
            return m_data;
        }

      private:
        data_array_t m_data;
        strides_cached(strides_cached const &);
    };

    template <typename T>
    struct is_strides_cached : boost::mpl::false_ {};

    template <uint_t ID, typename StorageInfoList>
    struct is_strides_cached<strides_cached<ID, StorageInfoList>> : boost::mpl::true_ {};

    namespace _impl {
        template <uint_t Coordinate, class LayoutMap, int Mapped = LayoutMap::template at_unsafe<Coordinate>()>
        struct is_dummy_coordinate : bool_constant<(Mapped < 0)> {};

        template <class StorageInfo, class LocalDomain>
        struct get_index : meta::st_position<typename LocalDomain::storage_info_ptr_list, StorageInfo const *> {};

        template <uint_t Coordinate,
            class LayoutMap,
            uint_t I,
            class Strides,
            // work-arround for gcc7 (force compile-time evaluation)
            int Cur = std::integral_constant<int, LayoutMap::template at_unsafe<Coordinate>()>::value,
            // work-arround for gcc7 (force compile-time evaluation)
            int Max = std::integral_constant<int, LayoutMap::max()>::value,
            enable_if_t<Cur<0, int> = 0> GT_FUNCTION int_t get_stride(Strides const &) {
            return 0;
        }

        template <uint_t Coordinate,
            class LayoutMap,
            uint_t I,
            class Strides,
            // work-arround for gcc7 (force compile-time evaluation)
            int Cur = std::integral_constant<int, LayoutMap::template at_unsafe<Coordinate>()>::value,
            // work-arround for gcc7 (force compile-time evaluation)
            int Max = std::integral_constant<int, LayoutMap::max()>::value,
            enable_if_t<Cur >= 0 && Cur == Max, int> = 0>
        GT_FUNCTION int_t get_stride(Strides const &) {
            return 1;
        }

        template <uint_t Coordinate,
            class LayoutMap,
            uint_t I,
            class Strides,
            // work-arround for gcc7 (force compile-time evaluation)
            int Cur = std::integral_constant<int, LayoutMap::template at_unsafe<Coordinate>()>::value,
            // work-arround for gcc7 (force compile-time evaluation)
            int Max = std::integral_constant<int, LayoutMap::max()>::value,
            enable_if_t<Cur >= 0 && Cur != Max, int> = 0>
        GT_FUNCTION int_t get_stride(Strides const &RESTRICT strides) {
            return strides.template get<I>()[Cur];
        }
    } // namespace _impl

    /**@brief incrementing all the storage pointers to the m_data_pointers array

       @tparam Coordinate direction along which the increment takes place
       @tparam StridesCached strides cached type

           This method is responsible of incrementing the index for the memory access at
           the location (i,j,k) incremented/decremented by 1 along the 'Coordinate' direction. Such index is shared
           among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.
    */
    template <typename LocalDomain, uint_t Coordinate, typename StridesCached, typename ArrayIndex>
    struct increment_index_functor {
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<StridesCached>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_array_of<ArrayIndex, int>::value), GT_INTERNAL_ERROR);

        const int_t m_increment;
        ArrayIndex &RESTRICT m_index_array;
        StridesCached const &RESTRICT m_strides_cached;

        template <typename StorageInfo,
            typename Layout = typename StorageInfo::layout_t,
            enable_if_t<_impl::is_dummy_coordinate<Coordinate, Layout>::value, int> = 0>
        GT_FUNCTION void operator()(const StorageInfo *) const {}

        template <typename StorageInfo,
            size_t I = _impl::get_index<StorageInfo, LocalDomain>::value,
            typename Layout = typename StorageInfo::layout_t,
            enable_if_t<!_impl::is_dummy_coordinate<Coordinate, Layout>::value, int> = 0>
        GT_FUNCTION void operator()(const StorageInfo *) const {
            GRIDTOOLS_STATIC_ASSERT(I < ArrayIndex::size(), "Accessing an index out of bound in fusion tuple");
            m_index_array[I] += _impl::get_stride<Coordinate, Layout, I>(m_strides_cached) * m_increment;
        }
    };

    template <uint_t Coordinate, class LocalDomain, class Strides, class ArrayIndex>
    GT_FUNCTION void do_increment(
        int_t step, LocalDomain const &local_domain, Strides const &RESTRICT strides, ArrayIndex &index) {
        boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
            increment_index_functor<LocalDomain, Coordinate, Strides, ArrayIndex>{step, index, strides});
    }

    template <uint_t Coordinate, ptrdiff_t Step, class LocalDomain, class Strides, class ArrayIndex>
    GT_FUNCTION void do_increment(LocalDomain const &local_domain, Strides const &RESTRICT strides, ArrayIndex &index) {
        boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
            increment_index_functor<LocalDomain, Coordinate, Strides, ArrayIndex>{Step, index, strides});
    }

    /**@brief functor initializing the indices does the actual assignment
     *     This method is responsible of computing the index for the memory access at
     *     the location (i,j,k). Such index is shared among all the fields contained in the
     *     same storage class instance, and it is not shared among different storage instances.
     * @tparam Coordinate direction along which the increment takes place
     * @tparam StridesCached strides cached type
     * @tparam StorageSequence sequence of storages
     */
    template <class StorageInfo, class MaxExtent, bool IsTmp>
    struct get_index_offset_f;

    template <class StorageInfo, class MaxExtent>
    struct get_index_offset_f<StorageInfo, MaxExtent, false> {
        template <class Backend, class Stride, class Begin, class BlockNo, class PosInBlock>
        GT_FUNCTION int_t operator()(Backend const &,
            Stride const &RESTRICT stride,
            Begin const &RESTRICT begin,
            BlockNo const &RESTRICT block_no,
            PosInBlock const &RESTRICT pos_in_block) const {
            static constexpr auto block_size =
                make_pos3(block_i_size(Backend{}), block_j_size(Backend{}), block_k_size(Backend{}));
            return stride.i * (begin.i + block_no.i * block_size.i + pos_in_block.i) +
                   stride.j * (begin.j + block_no.j * block_size.j + pos_in_block.j) +
                   stride.k * (begin.k + block_no.k * block_size.k + pos_in_block.k);
        }
    };

    template <class StorageInfo, class MaxExtent>
    struct get_index_offset_f<StorageInfo, MaxExtent, true> {
        template <class Backend, class Stride, class Begin, class BlockNo, class PosInBlock>
        GT_FUNCTION int_t operator()(Backend const &backend,
            Stride const &RESTRICT stride,
            Begin const &RESTRICT /*begin*/,
            BlockNo const &RESTRICT block_no,
            PosInBlock const &RESTRICT pos_in_block) const {
            return get_tmp_storage_offset<StorageInfo, MaxExtent>(backend, stride, block_no, pos_in_block);
        }
    };

    template <class Strides, class LocalDomain, class ArrayIndex, class Backend>
    struct initialize_index_f {
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached<Strides>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_array_of<ArrayIndex, int>::value), GT_INTERNAL_ERROR);
        Strides const &RESTRICT m_strides;
        pos3<uint_t> m_begin;
        pos3<uint_t> m_block_no;
        pos3<int_t> m_pos_in_block;
        ArrayIndex &RESTRICT m_index_array;

        template <typename StorageInfo, size_t I = _impl::get_index<StorageInfo, LocalDomain>::value>
        GT_FUNCTION void operator()(const StorageInfo *) const {
            GRIDTOOLS_STATIC_ASSERT(I < ArrayIndex::size(), "Accessing an index out of bound in fusion tuple");
            using max_extent_t = typename LocalDomain::max_extent_for_tmp_t;
            using layout_t = typename StorageInfo::layout_t;
            static constexpr auto backend = Backend{};
            static constexpr auto is_tmp =
                meta::st_contains<typename LocalDomain::tmp_storage_info_ptr_list, StorageInfo const *>::value;
            m_index_array[I] = get_index_offset_f<StorageInfo, max_extent_t, is_tmp>{}(backend,
                make_pos3(_impl::get_stride<coord_i<Backend>::value, layout_t, I>(m_strides),
                    _impl::get_stride<coord_j<Backend>::value, layout_t, I>(m_strides),
                    _impl::get_stride<coord_k<Backend>::value, layout_t, I>(m_strides)),
                m_begin,
                m_block_no,
                m_pos_in_block);
        }
    };

    /**@brief functor assigning the strides to a lobal array (i.e. m_strides).

       It implements the unrolling of a double loop: i.e. is n_f is the number of fields in this user function,
       and n_d(i) is the number of space dimensions per field (dependent on the ith field), then the loop for assigning
       the strides
       would look like
       for(i=0; i<n_f; ++i)
       for(j=0; j<n_d(i); ++j)
       * @tparam BackendType the type of backend
       * @tparam StridesCached strides cached type
       * @tparam LocalDomain local domain type
       */
    template <typename BackendType, typename StridesCached, typename LocalDomain>
    struct assign_strides {
        GRIDTOOLS_STATIC_ASSERT(is_strides_cached<StridesCached>::value, GT_INTERNAL_ERROR);

        template <typename SInfo>
        struct assign {
            const SInfo *m_storage_info;
            StridesCached &RESTRICT m_strides_cached;

            GT_FUNCTION assign(const SInfo *storage_info, StridesCached &RESTRICT strides_cached)
                : m_storage_info(storage_info), m_strides_cached(strides_cached) {}

            template <typename Coordinate>
            GT_FUNCTION enable_if_t<(Coordinate::value >= SInfo::layout_t::unmasked_length)> operator()() const {}

            template <typename Coordinate>
            GT_FUNCTION enable_if_t<(Coordinate::value < SInfo::layout_t::unmasked_length)> operator()() const {
                typedef typename SInfo::layout_t layout_map_t;
                using index_t = meta::st_position<typename LocalDomain::storage_info_ptr_list, SInfo const *>;
                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::contains<typename LocalDomain::storage_info_ptr_list, const SInfo *>::value),
                    GT_INTERNAL_ERROR_MSG(
                        "Error when trying to assign the strides in iterate domain. Access out of bounds."));
                constexpr int pos = SInfo::layout_t::template find<Coordinate::value>();
                GRIDTOOLS_STATIC_ASSERT(pos < SInfo::layout_t::masked_length,
                    GT_INTERNAL_ERROR_MSG(
                        "Error when trying to assign the strides in iterate domain. Access out of bounds."));

                BackendType::template once_per_block<index_t::value>::assign(
                    (m_strides_cached.template get<index_t::value>())[Coordinate::value],
                    m_storage_info->template stride<pos>());
            }
        };

        StridesCached &RESTRICT m_strides_cached;

        template <typename StorageInfo>
        GT_FUNCTION enable_if_t<StorageInfo::layout_t::unmasked_length == 0> operator()(StorageInfo const *) const {}

        template <typename StorageInfo>
        GT_FUNCTION enable_if_t<StorageInfo::layout_t::unmasked_length != 0> operator()(
            StorageInfo const *storage_info) const {
            using range = GT_META_CALL(meta::make_indices_c, StorageInfo::layout_t::unmasked_length - 1);
            host_device::for_each_type<range>(assign<StorageInfo>(storage_info, m_strides_cached));
        }
    };

    /**
     * function that checks a given pointer and offset combination results in an out of bounds access.
     * the check is computing the fields offset in order to get the base address of the accessed storage.
     * once the base address is known it can be checked if the requested access lies within the
     * storages allocated memory.
     */
    template <typename StorageInfo, typename LocalDomain>
    GT_FUNCTION bool pointer_oob_check(LocalDomain const &local_domain, int_t offset) {
        constexpr auto storage_info_index =
            meta::st_position<typename LocalDomain::storage_info_ptr_list, StorageInfo const *>::value;
        return offset < get<storage_info_index>(local_domain.m_local_padded_total_lengths) && offset >= 0;
    }

    /**
     * metafunction that evaluates if an accessor is cached by the backend
     * the Accessor parameter is either an Accessor or an expressions
     */
    template <size_t Index, class CachesMap>
    struct index_is_cached : boost::mpl::has_key<CachesMap, static_uint<Index>> {};

    template <class Arg, enumtype::intent Intent>
    struct deref_type : std::add_lvalue_reference<typename Arg::data_store_t::data_t> {};

    template <class Arg>
    struct deref_type<Arg, enumtype::in> {
        using type = typename Arg::data_store_t::data_t;
    };
} // namespace gridtools
