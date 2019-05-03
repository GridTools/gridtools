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

#include "../../common/defs.hpp"
#include "../../common/dimension.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "../is_accessor.hpp"

namespace gridtools {

    template <uint_t, int_t>
    struct pair_;

    /**@brief same as accessor but mixing run-time offsets with compile-time ones

       Method get() checks before among the static dimensions, and if the
       queried dimension is not found it looks up in the dynamic dimensions. Note that this
       lookup is anyway done at compile time, i.e. the get() method returns in constant time.

       FIXME(anstaf): at a moment accessor_mixed doesn't do compile time access. should be reimplemented using tuple
       with integral_constants and ints
     */
    template <class Base, class... Pairs>
    struct accessor_mixed;

    template <class Base, uint_t... Inxs, int_t... Vals>
    struct accessor_mixed<Base, pair_<Inxs, Vals>...> : Base {
        template <class... Ts>
        GT_FUNCTION explicit GT_CONSTEXPR accessor_mixed(Ts... args) : Base(dimension<Inxs>(Vals)..., args...) {}
    };

    /**
       @brief this struct allows the specification of SOME of the arguments before instantiating the offset_tuple.
       It is a language keyword. Usage examples can be found in the unit tests.
       Possible interfaces:
       - runtime alias
\verbatim
alias<arg_t, dimension<3> > field1(-3); //records the offset -3 as dynamic value
\endverbatim
       field1(args...) is then equivalent to arg_t(dimension<3>(-3), args...)
       - compiletime alias
\verbatim
        using field1 = alias<arg_t, dimension<7> >::set<-3>;
\endverbatim
       field1(args...) is then equivalent to arg_t(dimension<7>(-3), args...)

       NOTE: noone checks that you did not specify the same dimension twice. If that happens, the first occurrence of
the dimension is chosen
    */
    template <typename AccessorType, typename... Known>
    struct alias;

    template <typename AccessorType, uint_t... Inxs>
    struct alias<AccessorType, dimension<Inxs>...> {
        GT_STATIC_ASSERT(is_accessor<AccessorType>::value,
            "wrong type. If you want to generalize the alias "
            "to something more generic than an offset_tuple "
            "remove this assert.");

        /**
           @brief compile-time aliases, the offsets specified in this way are assured to be compile-time

           This type alias allows to embed some of the offsets directly inside the type of the accessor placeholder.
           For a usage example check the examples folder
        */
        template <int_t... Args>
        using set = accessor_mixed<AccessorType, pair_<Inxs, Args>...>;
    };

    template <typename... Types>
    struct is_accessor<accessor_mixed<Types...>> : std::true_type {};
} // namespace gridtools
