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

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "defs.hpp"
#include "generic_metafunctions/utility.hpp"

namespace gridtools {

    struct bad_any_cast : std::bad_cast {
        const char *what() const noexcept override { return "gridtools::bad_any_cast"; }
    };

    /**
     *  this class implements the subset of std::any interface and can hold move only objects.
     *
     *  TODO(anstaf): implement missing std::any components: piecewise ctors, emplace, reset, swap, make_any
     */
    class any_moveable {
        struct iface {
            virtual ~iface() = default;
            virtual std::type_info const &type() const noexcept = 0;
        };
        template <class T>
        struct impl : iface {
            T m_obj;
            impl(T const &obj) : m_obj(obj) {}
            impl(T &&obj) : m_obj(std::move(obj)) {}
            std::type_info const &type() const noexcept override { return typeid(T); }
        };
        std::unique_ptr<iface> m_impl;

      public:
        any_moveable() = default;

        template <class Arg, class Decayed = std::decay_t<Arg>>
        any_moveable(Arg &&arg) : m_impl(new impl<Decayed>(std::forward<Arg>(arg))) {}
        any_moveable(any_moveable &&) = default;

        template <class Arg, class Decayed = std::decay_t<Arg>>
        any_moveable &operator=(Arg &&obj) {
            m_impl.reset(new impl<Decayed>(std::forward<Arg>(obj)));
            return *this;
        }
        any_moveable &operator=(any_moveable &&) = default;

        bool has_value() const noexcept { return !!m_impl; }
        std::type_info const &type() const noexcept { return m_impl->type(); }

        template <class T>
        friend T *any_cast(any_moveable *src) noexcept {
            return src && src->type() == typeid(T) ? &static_cast<impl<T> *>(src->m_impl.get())->m_obj : nullptr;
        }
    };

    template <class T>
    T const *any_cast(any_moveable const *src) noexcept {
        return any_cast<T>(const_cast<any_moveable *>(src));
    }

    template <class T>
    T any_cast(any_moveable &src) {
        auto *ptr = any_cast<std::remove_reference_t<T>>(&src);
        if (!ptr)
            throw bad_any_cast{};
        using ref_t = std::conditional_t<std::is_reference<T>::value, T, std::add_lvalue_reference_t<T>>;
        return static_cast<ref_t>(*ptr);
    }

    template <class T>
    T any_cast(any_moveable const &src) {
        return any_cast<T>(const_cast<any_moveable &>(src));
    }

    template <class T>
    T any_cast(any_moveable &&src) {
        GT_STATIC_ASSERT(std::is_rvalue_reference<T &&>::value || std::is_const<std::remove_reference_t<T>>::value,
            "any_cast shall not be used for getting nonconst references to temporary objects");
        return any_cast<T>(src);
    }
} // namespace gridtools
