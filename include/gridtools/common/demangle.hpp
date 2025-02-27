// Vendored and slightly simplified version of boost::core::demangle
//
// Copyright 2014 Peter Dimov
// Copyright 2014 Andrey Semashev
//
// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt

#pragma once

#include <string>

#if __has_include(<cxxabi.h>)
#define GT_HAS_CXXABI_H
#endif

#if defined(GT_HAS_CXXABI_H)
#include <cxxabi.h>
#endif

namespace gridtools {

    namespace common {

        inline char const *demangle_alloc(char const *name) noexcept;
        inline void demangle_free(char const *name) noexcept;

        class scoped_demangled_name {
          private:
            char const *m_p;

          public:
            explicit scoped_demangled_name(char const *name) noexcept : m_p(demangle_alloc(name)) {}

            ~scoped_demangled_name() noexcept { demangle_free(m_p); }

            char const *get() const noexcept { return m_p; }

            scoped_demangled_name(scoped_demangled_name const &) = delete;
            scoped_demangled_name &operator=(scoped_demangled_name const &) = delete;
        };

#if defined(GT_HAS_CXXABI_H)

        inline char const *demangle_alloc(char const *name) noexcept {
            int status = 0;
            std::size_t size = 0;
            return abi::__cxa_demangle(name, NULL, &size, &status);
        }

        inline void demangle_free(char const *name) noexcept { std::free(const_cast<char *>(name)); }

        inline std::string demangle(char const *name) {
            scoped_demangled_name demangled_name(name);
            char const *p = demangled_name.get();
            if (!p)
                p = name;
            return p;
        }

#else

        inline char const *demangle_alloc(char const *name) noexcept { return name; }

        inline void demangle_free(char const *) noexcept {}

        inline std::string demangle(char const *name) { return name; }

#endif

    } // namespace common

} // namespace gridtools

#undef GT_HAS_CXXABI_H
