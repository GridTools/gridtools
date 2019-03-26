/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/gt_assert.hpp"
#include "../common/host_device.hpp"
#include "../meta/type_traits.hpp"
#include "common/definitions.hpp"
#include "data_store.hpp"

#ifndef GT_CHECK_MEMORY_SPACE

#ifdef __CUDA_ARCH__
#define GT_CHECK_MEMORY_SPACE(device_view) \
    GT_ASSERT_OR_THROW(device_view, "can not access a host view from within a GPU kernel")
#else
#define GT_CHECK_MEMORY_SPACE(device_view) \
    GT_ASSERT_OR_THROW(!device_view, "can not access a device view from a host function")
#endif

#endif

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /**
     * @brief data_view implementation. This struct provides means to modify contents of
     * gridtools data_store containers on arbitrary locations (host, device, etc.).
     * @tparam DataStore data store type
     * @tparam AccessMode access mode (default is read-write)
     */
    template <typename DataStore, access_mode AccessMode = access_mode::read_write>
    struct data_view {
        GT_STATIC_ASSERT(is_data_store<DataStore>::value, GT_INTERNAL_ERROR_MSG("Passed type is no data_store type"));
        using data_store_t = DataStore;
        typedef typename DataStore::data_t data_t;
        typedef typename DataStore::state_machine_t state_machine_t;
        typedef typename DataStore::storage_info_t storage_info_t;
        const static access_mode mode = AccessMode;

      private:
        data_t *m_raw_ptr;

        state_machine_t *m_state_machine_ptr;
        storage_info_t const *m_storage_info;
        bool m_device_view;

      public:
        /**
         * @brief data_view constructor
         */
        GT_FUNCTION data_view()
            : m_raw_ptr(nullptr), m_state_machine_ptr(nullptr), m_storage_info(nullptr), m_device_view(false) {}

        /**
         * @brief data_view constructor. This constructor is normally not called by the user because it is more
         * convenient to use the provided make functions.
         * @param data_ptr pointer to the data
         * @param info_ptr pointer to the storage_info
         * @param state_ptr pointer to the state machine
         * @param device_view true if device view, false otherwise
         */
        GT_FUNCTION data_view(
            data_t *data_ptr, storage_info_t const *info_ptr, state_machine_t *state_ptr, bool device_view)
            : m_raw_ptr(data_ptr), m_state_machine_ptr(state_ptr), m_storage_info(info_ptr),
              m_device_view(device_view) {
            GT_ASSERT_OR_THROW(data_ptr, "Cannot create data_view with invalid data pointer");
            GT_ASSERT_OR_THROW(info_ptr, "Cannot create data_view with invalid storage info pointer");
        }

        storage_info_t const &storage_info() const {
            GT_CHECK_MEMORY_SPACE(m_device_view);
            return *m_storage_info;
        }

        /**
         * data getter
         */
        GT_FUNCTION
        data_t *data() { return m_raw_ptr; }

        /**
         * data getter
         */
        GT_FUNCTION
        data_t const *data() const { return m_raw_ptr; }

        /**
         * @return pointer to the first position
         */
        GT_FUNCTION
        data_t *ptr_to_first_position() { return &operator()(gridtools::array<int, storage_info_t::ndims>{{}}); }

        /**
         * return pointer to the first position
         */
        GT_FUNCTION
        data_t const *ptr_to_first_position() const {
            return &operator()(gridtools::array<int, storage_info_t::ndims>{{}});
        }

        /**
         * @brief operator() is used to access elements. E.g., view(0,0,2) will return the third element.
         * @param c given indices
         * @return reference to the queried value
         */
        template <typename... Coords>
        conditional_t<AccessMode == access_mode::read_only, data_t const &, data_t &> GT_FUNCTION operator()(
            Coords... c) const {
            GT_STATIC_ASSERT(conjunction<is_all_integral_or_enum<Coords...>>::value,
                GT_INTERNAL_ERROR_MSG("Index arguments have to be integral types."));
            GT_CHECK_MEMORY_SPACE(m_device_view);
            return m_raw_ptr[m_storage_info->index(c...)];
        }

        /**
         * @brief operator() is used to access elements. E.g., view({0,0,2}) will return the third element.
         * @param arr array of indices
         * @return reference to the queried value
         */
        conditional_t<AccessMode == access_mode::read_only, data_t const &, data_t &> GT_FUNCTION operator()(
            gridtools::array<int, storage_info_t::ndims> const &arr) const {
            GT_CHECK_MEMORY_SPACE(m_device_view);
            return m_raw_ptr[m_storage_info->index(arr)];
        }

        /**
         * @brief Check if view contains valid pointers, and simple state machine checks.
         * Be aware that this is not a full check. In order to check if a view is in a
         * consistent state use check_consistency function.
         * @return true if pointers and state is correct, otherwise false
         */
        bool valid() const {
            // ptrs invalid -> view invalid
            if (!m_raw_ptr || !m_storage_info)
                return false;
            // when used in combination with a host storage the view is always valid as long as the ptrs are
            if (!m_state_machine_ptr)
                return true;
            // read only -> simple check
            if (AccessMode == access_mode::read_only)
                return m_device_view ? !m_state_machine_ptr->m_dnu : !m_state_machine_ptr->m_hnu;
            else
                // check state machine ptrs
                return m_device_view ? ((m_state_machine_ptr->m_hnu) && !(m_state_machine_ptr->m_dnu))
                                     : (!(m_state_machine_ptr->m_hnu) && (m_state_machine_ptr->m_dnu));
        }

        /*
         * @brief member function to retrieve the total size (dimensions, halos, padding, initial_offset).
         * @return total size
         */
        GT_FUNCTION constexpr int padded_total_length() const { return m_storage_info->padded_total_length(); }

        /*
         * @brief Returns the length of a dimension excluding the halo points (only the inner region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr int length() const {
            return m_storage_info->template length<Dim>();
        }

        /*
         * @brief Returns the length of a dimension including the halo points (the outer region)
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr int total_length() const {
            return m_storage_info->template total_length<Dim>();
        }

        /*
         * @brief Returns the index of the first element in the specified dimension when iterating in the whole outer
         * region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr int total_begin() const {
            return m_storage_info->template total_begin<Dim>();
        }

        /*
         * @brief Returns the index of the first element in the specified dimension when iterating in the inner region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr int begin() const {
            return m_storage_info->template begin<Dim>();
        }

        /*
         * @brief Returns the index of the last element in the specified dimension when iterating in the whole outer
         * region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr int total_end() const {
            return m_storage_info->template total_end<Dim>();
        }

        /*
         * @brief Returns the index of the last element in the specified dimension when iterating in the inner region
         *
         * \tparam Dim The index of the dimension
         */
        template <uint_t Dim>
        GT_FUNCTION constexpr int end() const {
            return m_storage_info->template end<Dim>();
        }

        friend data_t *advanced_get_raw_pointer_of(data_view const &src) { return src.m_raw_ptr; }
    };

    template <typename T>
    struct is_data_view : std::false_type {};

    template <typename Storage, access_mode AccessMode>
    struct is_data_view<data_view<Storage, AccessMode>> : std::true_type {};

    namespace advanced {
        /** Function to access the protected data member of the views
            containing the raw pointers.  This is an interface we want to avoid
            using (and future fixed of the communication library will not need
            this. We made the use of this function difficult on purpose.
        */
        template <class T>
        auto get_raw_pointer_of(T const &src) GT_AUTO_RETURN(advanced_get_raw_pointer_of(src));
    } // namespace advanced

    /**
     * @}
     */
} // namespace gridtools

#undef GT_CHECK_MEMORY_SPACE
