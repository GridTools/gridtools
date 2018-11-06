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

#include <type_traits>

#include "../../common/generic_metafunctions/type_traits.hpp"
#include "../accessor_metafunctions.hpp"
#include "../expressions/expressions.hpp"
#include "accessor.hpp"

namespace gridtools {

    template <typename T>
    struct is_regular_accessor : boost::mpl::false_ {};

    template <uint_t ID, enumtype::intent Intent, typename Extent, ushort_t Number>
    struct is_regular_accessor<accessor<ID, Intent, Extent, Number>> : boost::mpl::true_ {};

    template <typename T>
    struct is_regular_accessor<const T> : is_regular_accessor<T> {};

    template <typename T>
    struct is_accessor : boost::mpl::false_ {};

    template <uint_t ID, enumtype::intent Intent, typename Extent, ushort_t Number>
    struct is_accessor<accessor<ID, Intent, Extent, Number>> : boost::mpl::true_ {};

    template <typename T>
    struct is_grid_accessor : boost::mpl::false_ {};

    template <uint_t ID, enumtype::intent Intent, typename Extent, ushort_t Number>
    struct is_grid_accessor<accessor<ID, Intent, Extent, Number>> : boost::mpl::true_ {};

    /**
     * @brief metafunction that given an accesor and a map, it will remap the index of the accessor according
     * to the corresponding entry in ArgsMap
     */
    template <typename Accessor, typename ArgsMap, typename Enable = void>
    struct remap_accessor_type {};

    // TODO(havogt) I have no idea why I end up here...
    template <typename ArgsMap>
    struct remap_accessor_type<boost::mpl::void_, ArgsMap> {
        using type = boost::mpl::void_;
    };

    template <ushort_t ID, enumtype::intent Intent, typename Extent, ushort_t Number, typename ArgsMap>
    struct remap_accessor_type<accessor<ID, Intent, Extent, Number>, ArgsMap> {
        using type = accessor<_impl::get_remap_accessor_id<ID, ArgsMap>(), Intent, Extent, Number>;
    };

    template <class ArgsMap, class Op, class... Args>
    struct remap_accessor_type<expr<Op, Args...>, ArgsMap> {
        // Here we traverse the expression AST down to the leaves, and we assert if
        // the leaves are not accessor types.

        // recursively remapping the template arguments,
        // until the specialization above stops the recursion
        using type = expr<Op, typename remap_accessor_type<Args, ArgsMap>::type...>;
    };

    template <typename T, typename ArgsMap>
    struct remap_accessor_type<T, ArgsMap, enable_if_t<std::is_arithmetic<T>::value>> {
        // when a leaf don't do anything
        typedef T type;
    };

} // namespace gridtools
