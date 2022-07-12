/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <pybind11/pybind11.h>

#include "../common/array.hpp"
#include "neighbor_table.hpp"

namespace gridtools::fn {
    namespace python_neighbor_table_adapter_impl_ {
        template <class T, std::size_t MaxNeighbors>
        struct wrapper {
            pybind11::buffer_info m_info;

            friend array<T, MaxNeighbors> neighbor_table_neighbors(wrapper const &obj, int index) {
                int *ptr = reinterpret_cast<T *>(obj.m_info.ptr);
                ptr += index * obj.m_info.strides[0] / obj.m_info.itemsize;
                // auto val = *ptr;
                // ptr += obj.m_info.strides[1] / obj.m_info.itemsize;
                printf("s0: %d\n", obj.m_info.strides[0]);
                printf("s1: %d\n", obj.m_info.strides[1]);
                return {ptr[0], ptr[1]};
            };
        };
        template <class T, std::size_t MaxNeighbors>
        wrapper<T, MaxNeighbors> as_neighbor_table(pybind11::buffer src) {
            auto info = src.request(false);
            return {std::move(info)};
        }

    } // namespace python_neighbor_table_adapter_impl_

    using python_neighbor_table_adapter_impl_::as_neighbor_table;
} // namespace gridtools::fn
