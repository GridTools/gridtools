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

#include <array>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <pybind11/pybind11.h>

#include "../../common/integral_constant.hpp"
#include "../../sid/simple_ptr_holder.hpp"

namespace gridtools {
    namespace python_sid_adapter_impl_ {
        template <size_t, class>
        struct kind {};

        template <class T, size_t Dim, class Kind>
        struct wrapper {
            pybind11::buffer_info m_info;

            friend sid::simple_ptr_holder<T *> sid_get_origin(wrapper const &obj) {
                return {reinterpret_cast<T *>(obj.m_info.ptr)};
            }
            friend std::array<pybind11::ssize_t, Dim> sid_get_strides(wrapper const &obj) {
                std::array<pybind11::ssize_t, Dim> res;
                assert(obj.m_info.strides.size() == Dim);
                for (std::size_t i = 0; i != Dim; ++i) {
                    assert(obj.m_info.strides[i] % obj.m_info.itemsize == 0);
                    res[i] = obj.m_info.strides[i] / obj.m_info.itemsize;
                }
                return res;
            }
            friend std::array<integral_constant<pybind11::ssize_t, 0>, Dim> sid_get_lower_bounds(wrapper const &) {
                return {};
            }
            friend std::array<pybind11::ssize_t, Dim> sid_get_upper_bounds(wrapper const &obj) {
                std::array<pybind11::ssize_t, Dim> res;
                assert(obj.m_info.shape.size() == Dim);
                for (std::size_t i = 0; i != Dim; ++i) {
                    assert(obj.m_info.shape[i] > 0);
                    res[i] = obj.m_info.shape[i];
                }
                return res;
            }
            friend kind<Dim, Kind> sid_get_strides_kind(wrapper const &) { return {}; }
        };

        template <class T, std::size_t Dim, class Kind = void>
        wrapper<T, Dim, Kind> as_sid(pybind11::buffer const &src) {
            static_assert(
                std::is_trivially_copyable<T>::value, "as_sid should be instantiated with the trivially copyable type");
            constexpr bool writable = !std::is_const<T>();
            // pybind11::buffer::request accepts writable as an optional parameter (default is false).
            // if writable is true PyBUF_WRITABLE flag is added while delegating to the PyObject_GetBuffer.
            auto info = src.request(writable);
            assert(!(writable && info.readonly));
            if (info.ndim != Dim)
                throw std::domain_error("buffer has incorrect number of dimensions: " + std::to_string(info.ndim) +
                                        "; expected " + std::to_string(Dim));
            if (info.itemsize != sizeof(T))
                throw std::domain_error("buffer has incorrect itemsize: " + std::to_string(info.itemsize) +
                                        "; expected " + std::to_string(sizeof(T)));
            using format_desc_t = pybind11::format_descriptor<std::remove_const_t<T>>;
            if (info.format != format_desc_t::format())
                throw std::domain_error(
                    "buffer has incorrect format: " + info.format + "; expected " + format_desc_t::format());
            return {std::move(info)};
        }
    } // namespace python_sid_adapter_impl_

    // Makes a SID from the `pybind11::buffer`.
    // Be aware that the return value is a move only object
    using python_sid_adapter_impl_::as_sid;
} // namespace gridtools
