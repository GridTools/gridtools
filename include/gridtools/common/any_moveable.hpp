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

#include <memory>
#include <typeinfo>
#include <type_traits>
#include <utility>

#include "defs.hpp"

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
        template < class T >
        struct impl : iface {
            T m_obj;
            impl(T const &obj) : m_obj(obj) {}
            impl(T &&obj) : m_obj(std::move(obj)) {}
            std::type_info const &type() const noexcept override { return typeid(T); }
        };
        std::unique_ptr< iface > m_impl;

      public:
        any_moveable() = default;

        template < class Arg, class Decayed = typename std::decay< Arg >::type >
        any_moveable(Arg &&arg)
            : m_impl(new impl< Decayed >(std::forward< Arg >(arg))) {}
        any_moveable(any_moveable &&) = default;

        template < class Arg, class Decayed = typename std::decay< Arg >::type >
        any_moveable &operator=(Arg &&obj) {
            m_impl.reset(new impl< Decayed >(std::forward< Arg >(obj)));
            return *this;
        }
        any_moveable &operator=(any_moveable &&) = default;

        bool has_value() const noexcept { return !!m_impl; }
        std::type_info const &type() const noexcept { return m_impl->type(); }

        template < class T >
        friend T *any_cast(any_moveable *src) noexcept {
            return src && src->type() == typeid(T) ? &static_cast< impl< T > * >(src->m_impl.get())->m_obj : nullptr;
        }
    };

    template < class T >
    T const *any_cast(any_moveable const *src) noexcept {
        return any_cast< T >(const_cast< any_moveable * >(src));
    }

    template < class T >
    T any_cast(any_moveable &src) {
        auto *ptr = any_cast< typename std::remove_reference< T >::type >(&src);
        if (!ptr)
            throw bad_any_cast{};
        using ref_t = typename std::conditional< std::is_reference< T >::value,
            T,
            typename std::add_lvalue_reference< T >::type >::type;
        return static_cast< ref_t >(*ptr);
    }

    template < class T >
    T any_cast(any_moveable const &src) {
        return any_cast< T >(const_cast< any_moveable & >(src));
    }

    template < class T >
    T any_cast(any_moveable &&src) {
        GRIDTOOLS_STATIC_ASSERT(std::is_rvalue_reference< T && >::value ||
                                    std::is_const< typename std::remove_reference< T >::type >::value,
            "any_cast shall not be used for getting nonconst references to temporary objects");
        return any_cast< T >(src);
    }
}
