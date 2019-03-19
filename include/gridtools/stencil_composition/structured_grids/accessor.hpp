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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/always.hpp"
#include "../accessor_base.hpp"
#include "../accessor_intent.hpp"
#include "../is_accessor.hpp"
#include "extent.hpp"
/**
   @file

   @brief File containing the definition of the regular accessor used
   to address the storage (at offsets) from whithin the functors.
   This accessor is a proxy for a storage class, i.e. it is a light
   object used in place of the storage when defining the high level
   computations, and it will be bound later on with a specific
   instantiation of a storage class.

   An accessor can be instantiated directly in the apply
   method, or it might be a constant expression instantiated outside
   the functor scope and with static duration.
*/

namespace gridtools {

    /**
       @brief the definition of accessor visible to the user

       \tparam ID the integer unique ID of the field placeholder

       \tparam Extent the extent of i/j indices spanned by the
               placeholder, in the form of <i_minus, i_plus, j_minus,
               j_plus>.  The values are relative to the current
               position. See e.g. horizontal_diffusion::out_function
               as a usage example.

       \tparam Number the number of dimensions accessed by the
               field. Notice that we don't distinguish at this level what we
               call "space dimensions" from the "field dimensions". Both of
               them are accessed using the same interface. whether they are
               field dimensions or space dimension will be decided at the
               moment of the storage instantiation (in the main function)
     */
    template <uint_t ID, intent Intent = intent::in, typename Extent = extent<>, size_t Number = 3>
    struct accessor : accessor_base<Number> {
        using index_t = static_uint<ID>;
        static constexpr intent intent_v = Intent;
        using extent_t = Extent;

        /**inheriting all constructors from accessor_base*/
        using accessor_base<Number>::accessor_base;
    };

    template <uint_t ID, intent Intent, typename Extent, size_t Number>
    meta::always<accessor<ID, Intent, Extent, Number>> tuple_from_types(accessor<ID, Intent, Extent, Number> const &);

    template <uint_t ID, typename Extent = extent<>, size_t Number = 3>
    using in_accessor = accessor<ID, intent::in, Extent, Number>;

    template <uint_t ID, typename Extent = extent<>, size_t Number = 3>
    using inout_accessor = accessor<ID, intent::inout, Extent, Number>;

    template <uint_t ID, intent Intent, typename Extent, size_t Number>
    struct is_accessor<accessor<ID, Intent, Extent, Number>> : std::true_type {};

} // namespace gridtools
