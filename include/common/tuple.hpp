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
/**
 * Implementation of a tuple with constexpr ctr and random element getter
 * Similar to std::tuple but GPU capable.
 * Notice: Current version is optimized for stornig integrals elements, returning elements by constexpr copy.
 * In order to use with more complex types (non constexprable), a version returning by ref should be used instead
 */
#pragma once
#include "host_device.hpp"
#include "generic_metafunctions/variadic_typedef.hpp"

namespace gridtools {

    namespace _impl {
        template < ushort_t Idx, typename VariadicArgs, typename First, typename Super >
        struct return_helper {
            GT_FUNCTION constexpr typename VariadicArgs::template get_elem< Idx >::type operator()(
                const First f, const Super x) const {
                return x.template get< Idx - 1 >();
            }
        };

        template < typename VariadicArgs, typename First, typename Super >
        struct return_helper< 0, VariadicArgs, First, Super > {
            GT_FUNCTION constexpr First operator()(const First f, const Super x) const { return f; }
        };
    }

    template < typename... Args >
    struct tuple;

    template < typename ElementType, typename... OtherElements >
    struct tuple< ElementType, OtherElements... > : public tuple< OtherElements... > {

        typedef tuple< ElementType, OtherElements... > type;
        typedef variadic_typedef< ElementType, OtherElements... > tuple_elements_t;
        typedef tuple< OtherElements... > super;

        static const size_t n_dimensions = sizeof...(OtherElements) + 1;

        // ctr
        GT_FUNCTION constexpr tuple(const ElementType t, OtherElements const... x) : super(x...), m_elem(t) {}

        GT_FUNCTION constexpr ElementType operator()() const { return m_elem; }

        /**@brief returns the element at a specific index Idx*/
        template < ushort_t Idx >
        GT_FUNCTION constexpr typename tuple_elements_t::template get_elem< Idx >::type get() const {

            typedef _impl::return_helper< Idx, tuple_elements_t, ElementType, super > helper;
            return helper()(m_elem, *this);
        }

      protected:
        ElementType m_elem;
    };

    template < typename ElementType >
    struct tuple< ElementType > {

        static const size_t n_dimensions = 1;

        typedef tuple< ElementType > type;
        typedef variadic_typedef< ElementType > tuple_elements_t;

        template < ushort_t Idx >
        struct get_elem {
            typedef typename tuple_elements_t::template get_elem< Idx >::type type;
        };

        GT_FUNCTION constexpr tuple(const ElementType t) : m_elem(t) {}

        GT_FUNCTION constexpr ElementType operator()() const { return m_elem; }

        /**@brief returns the offset at a specific index Idx*/
        template < ushort_t Idx >
        GT_FUNCTION constexpr typename get_elem< Idx >::type get() const {
            GRIDTOOLS_STATIC_ASSERT((Idx == 0), "Error: out of bound tuple access");
            return m_elem;
        }

      protected:
        ElementType m_elem;
    };

    template < typename T >
    struct is_tuple : boost::mpl::false_ {};

    template < typename... Args >
    struct is_tuple< tuple< Args... > > : boost::mpl::true_ {};

    template < typename... Args >
    tuple< Args... > make_tuple(Args... args) {
        return tuple< Args... >(args...);
    }
} // namespace gridtools
