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

#include <utility>

#include <pybind11/pybind11.h>

#include "../common/array.hpp"
#include "neighbor_table.hpp"

namespace gridtools::fn {
    namespace python_neighbor_table_adapter_impl_ {
        template <class T, std::size_t... Is>
        array<T, sizeof...(Is)> array_maker(T *ptr, std::index_sequence<Is...>) {
            return {ptr[Is]...};
        }

        template <class T, std::size_t MaxNeighbors>
        struct wrapper {
            T *ptr;
            pybind11::ssize_t stride;

            friend array<T, MaxNeighbors> neighbor_table_neighbors(wrapper const &obj, int index) {
                int *ptr = reinterpret_cast<T *>(obj.ptr);
                ptr += index * obj.stride;
                return array_maker(ptr, std::make_index_sequence<MaxNeighbors>{});
            };
        };
        template <class T, std::size_t MaxNeighbors>
        wrapper<T, MaxNeighbors> as_neighbor_table(pybind11::buffer src) {
            auto info = src.request(false);
            // TODO check stride 1 dim etc.
            return {reinterpret_cast<T *>(info.ptr), (info.strides[0] / info.itemsize)};
        }

    } // namespace python_neighbor_table_adapter_impl_

    using python_neighbor_table_adapter_impl_::as_neighbor_table;
} // namespace gridtools::fn
