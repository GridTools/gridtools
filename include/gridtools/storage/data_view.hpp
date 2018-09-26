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

#include <assert.h>

#include <boost/mpl/and.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>

#include "../common/gt_assert.hpp"
#include "common/definitions.hpp"
#include "common/storage_info_interface.hpp"

#ifndef CHECK_MEMORY_SPACE

#ifdef __CUDA_ARCH__
#define CHECK_MEMORY_SPACE(device_view) \
    ASSERT_OR_THROW(device_view, "can not access a host view from within a GPU kernel")
#else
#define CHECK_MEMORY_SPACE(device_view) \
    ASSERT_OR_THROW(!device_view, "can not access a device view from a host function")
#endif

#endif

namespace gridtools {

    /** \ingroup storage
     * @{
     */
    template <typename DataStore, access_mode AccessMode = access_mode::ReadWrite>
    struct data_view;

    namespace advanced {
        /** Function to access the protected data member of the views
            containing the raw pointers.  This is an interface we want to avoid
            using (and future fixed of the communication library will not need
            this. We made the use of this function difficult on purpose.

            \tparam DataView The data_view type (deduced)

            \param dv The data_view object
            \param i The index of the pointer in the arrays of raw pointers
        */
        template <typename DataView>
        typename DataView::data_t *get_raw_pointer_of(DataView const &dv, int i = 0) {
            return dv.m_raw_ptrs[i];
        }

        template <typename DataStore, access_mode AccessMode>
        typename DataStore::storage_info_t const *storage_info_raw_ptr(data_view<DataStore, AccessMode> const &);

    } // namespace advanced

    /**
     * @brief data_view implementation. This struct provides means to modify contents of
     * gridtools data_store containers on arbitrary locations (host, device, etc.).
     * @tparam DataStore data store type
     * @tparam AccessMode access mode (default is read-write)
     */
    template <typename DataStore, access_mode AccessMode>
    struct data_view {
        GRIDTOOLS_STATIC_ASSERT(
            is_data_store<DataStore>::value, GT_INTERNAL_ERROR_MSG("Passed type is no data_store type"));
        using data_store_t = DataStore;
        typedef typename DataStore::data_t data_t;
        typedef typename DataStore::state_machine_t state_machine_t;
        typedef typename DataStore::storage_info_t storage_info_t;
        const static access_mode mode = AccessMode;
        const static uint_t num_of_storages = 1;

      private:
        data_t *m_raw_ptrs[1];

        state_machine_t *m_state_machine_ptr;
        storage_info_t const *m_storage_info;
        bool m_device_view;

      public:
        /**
         * @brief data_view constructor
         */
        GT_FUNCTION data_view()
            : m_raw_ptrs{NULL}, m_state_machine_ptr(NULL), m_storage_info(NULL), m_device_view(false) {}

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
            : m_raw_ptrs{data_ptr}, m_state_machine_ptr(state_ptr), m_storage_info(info_ptr),
              m_device_view(device_view) {
            ASSERT_OR_THROW(data_ptr, "Cannot create data_view with invalid data pointer");
            ASSERT_OR_THROW(info_ptr, "Cannot create data_view with invalid storage info pointer");
        }

        storage_info_t const &storage_info() const {
            CHECK_MEMORY_SPACE(m_device_view);
            return *m_storage_info;
        }

        /**
         * data getter
         */
        GT_FUNCTION
        data_t *data() { return m_raw_ptrs[0]; }

        /**
         * data getter
         */
        GT_FUNCTION
        data_t const *data() const { return m_raw_ptrs[0]; }

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
        typename boost::mpl::if_c<(AccessMode == access_mode::ReadOnly), data_t const &, data_t &>::type GT_FUNCTION
        operator()(Coords... c) const {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::and_<boost::mpl::bool_<(sizeof...(Coords) > 0)>,
                                        typename is_all_integral_or_enum<Coords...>::type>::value),
                GT_INTERNAL_ERROR_MSG("Index arguments have to be integral types."));
            CHECK_MEMORY_SPACE(m_device_view);
            return m_raw_ptrs[0][m_storage_info->index(c...)];
        }

        /**
         * @brief operator() is used to access elements. E.g., view({0,0,2}) will return the third element.
         * @param arr array of indices
         * @return reference to the queried value
         */
        typename boost::mpl::if_c<(AccessMode == access_mode::ReadOnly), data_t const &, data_t &>::type GT_FUNCTION
        operator()(gridtools::array<int, storage_info_t::ndims> const &arr) const {
            CHECK_MEMORY_SPACE(m_device_view);
            return m_raw_ptrs[0][m_storage_info->index(arr)];
        }

        /**
         * @brief Check if view contains valid pointers, and simple state machine checks.
         * Be aware that this is not a full check. In order to check if a view is in a
         * consistent state use check_consistency function.
         * @return true if pointers and state is correct, otherwise false
         */
        bool valid() const {
            // ptrs invalid -> view invalid
            if (!m_raw_ptrs[0] || !m_storage_info)
                return false;
            // when used in combination with a host storage the view is always valid as long as the ptrs are
            if (!m_state_machine_ptr)
                return true;
            // read only -> simple check
            if (AccessMode == access_mode::ReadOnly)
                return m_device_view ? !m_state_machine_ptr->m_dnu : !m_state_machine_ptr->m_hnu;
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

        template <typename T>
        friend typename T::data_t *advanced::get_raw_pointer_of(T const &, int);

        template <typename D, access_mode A>
        friend typename D::storage_info_t const *advanced::storage_info_raw_ptr(data_view<D, A> const &);
    };

    template <typename T>
    struct is_data_view : boost::mpl::false_ {};

    template <typename Storage, access_mode AccessMode>
    struct is_data_view<data_view<Storage, AccessMode>> : boost::mpl::true_ {};

    namespace advanced {
        template <typename DataStore, access_mode AccessMode>
        typename DataStore::storage_info_t const *storage_info_raw_ptr(data_view<DataStore, AccessMode> const &src) {
            return src.m_storage_info;
        }
    } // namespace advanced
    /**
     * @}
     */
} // namespace gridtools

#undef CHECK_MEMORY_SPACE
