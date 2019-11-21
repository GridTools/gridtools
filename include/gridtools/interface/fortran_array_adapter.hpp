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

#include "../storage/data_store.hpp"
#include "./layout_transformation/layout_transformation.hpp"
#include <cpp_bindgen/fortran_array_view.hpp>

namespace gridtools {
    template <class DataStore>
    class fortran_array_adapter {
        static_assert(storage::is_data_store<DataStore>::value, "");

      public:
        fortran_array_adapter(const bindgen_fortran_array_descriptor &descriptor) : m_descriptor(descriptor) {
            if (m_descriptor.rank != bindgen_view_rank::value)
                throw std::runtime_error("rank does not match (descriptor-rank [" + std::to_string(m_descriptor.rank) +
                                         "] != datastore-rank [" + std::to_string(bindgen_view_rank::value) + "]");
        }

        using bindgen_view_rank = std::integral_constant<size_t, DataStore::layout_t::unmasked_length>;
        using bindgen_view_element_type = typename DataStore::data_t;
        using bindgen_is_acc_present = bool_constant<true>;

        friend void transform(std::shared_ptr<DataStore> const &dest, fortran_array_adapter const &src) {
            adapter{const_cast<fortran_array_adapter &>(src), *dest}.from_array();
        }
        friend void transform(fortran_array_adapter &dest, std::shared_ptr<DataStore> const &src) {
            adapter{dest, *src}.to_array();
        }

      private:
        class adapter {
            using ElementType = typename DataStore::data_t;

          public:
            adapter(fortran_array_adapter &view, DataStore &data_store)
                : m_fortran_pointer(static_cast<ElementType *>(view.m_descriptor.data)),
                  m_cpp_pointer(data_store.get_target_ptr()), m_lengths(data_store.lengths()),
                  m_cpp_strides(data_store.strides()) {
                assert(m_fortran_pointer);

                // verify dimensions of fortran array
                for (uint_t c_dim = 0, fortran_dim = 0; c_dim < DataStore::layout_t::masked_length; ++c_dim)
                    if (m_cpp_strides[c_dim] != 0) {
                        if (view.m_descriptor.dims[fortran_dim] != m_lengths[c_dim])
                            throw std::runtime_error("dimensions do not match (descriptor [" +
                                                     std::to_string(view.m_descriptor.dims[fortran_dim]) +
                                                     "] != data_store [" + std::to_string(m_lengths[c_dim]) + "])");
                        ++fortran_dim;
                    }

                uint_t current_stride = 1;
                for (uint_t i = 0; i < m_fortran_strides.size(); ++i)
                    if (m_cpp_strides[i] != 0) {
                        m_fortran_strides[i] = current_stride;
                        current_stride *= m_lengths[i];
                    } else {
                        m_fortran_strides[i] = 0;
                    }
            }

            void from_array() const {
                interface::transform(m_cpp_pointer, m_fortran_pointer, m_lengths, m_cpp_strides, m_fortran_strides);
            }
            void to_array() const {
                interface::transform(m_fortran_pointer, m_cpp_pointer, m_lengths, m_fortran_strides, m_cpp_strides);
            }

          private:
            using lengths_t = std::decay_t<decltype(std::declval<DataStore const &>().lengths())>;
            using strides_t = std::decay_t<decltype(std::declval<DataStore const &>().strides())>;

            ElementType *m_fortran_pointer;
            ElementType *m_cpp_pointer;
            lengths_t m_lengths;
            strides_t m_fortran_strides;
            strides_t m_cpp_strides;
        };

        const bindgen_fortran_array_descriptor &m_descriptor;
    };
} // namespace gridtools
