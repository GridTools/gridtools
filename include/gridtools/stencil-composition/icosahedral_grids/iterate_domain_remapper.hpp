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
/*
  @file
  This file provides functionality for a iterate domain remapper that intercepts calls to iterate domain
  and remap the arguments to the actual positions in the iterate domain
*/

#pragma once

#include "../../common/defs.hpp"
#include "../../meta.hpp"
#include "../accessor.hpp"
#include "../arg.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../iterate_domain_metafunctions.hpp"
#include "./icosahedral_topology.hpp"
#include "./on_neighbors.hpp"

namespace gridtools {
    /**
     * @class iterate_domain_remapper
     * @param IterateDomain iterate domain
     * @param EsfArgsMap map from ESF arguments to iterate domain position of args.
     */
    template <typename IterateDomain, typename EsfArgs, typename EsfLocationType, uint_t Color>
    class iterate_domain_remapper {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_plh, EsfArgs>::value), GT_INTERNAL_ERROR);

        using domain_args_t = typename IterateDomain::esf_args_t;
        template <class Arg>
        GT_META_DEFINE_ALIAS(get_domain_index, meta::st_position, (domain_args_t, Arg));

        DISALLOW_COPY_AND_ASSIGN(iterate_domain_remapper);

        const IterateDomain &m_iterate_domain;

      public:
        typedef GT_META_CALL(meta::transform, (get_domain_index, EsfArgs)) esf_args_map_t;

        template <typename Accessor>
        using accessor_return_type = typename IterateDomain::template accessor_return_type<
            typename remap_accessor_type<Accessor, esf_args_map_t>::type>;

        GT_FUNCTION
        explicit iterate_domain_remapper(const IterateDomain &iterate_domain) : m_iterate_domain(iterate_domain) {}

        template <typename Accessor>
        GT_FUNCTION auto operator()(Accessor const &arg) const GT_AUTO_RETURN((
            m_iterate_domain(static_uint<Color>{}, typename remap_accessor_type<Accessor, esf_args_map_t>::type(arg))));

        template <typename ValueType, typename LocationTypeT, typename Reduction, typename... Accessors>
        GT_FUNCTION ValueType operator()(
            on_neighbors<ValueType, LocationTypeT, Reduction, Accessors...> onneighbors) const {
            constexpr auto offsets = connectivity<EsfLocationType, LocationTypeT, Color>::offsets();
            for (auto &&offset : offsets)
                onneighbors.m_value = onneighbors.m_function(
                    m_iterate_domain._evaluate(
                        typename remap_accessor_type<Accessors, esf_args_map_t>::type{}, offset)...,
                    onneighbors.m_value);
            return onneighbors.m_value;
        }
    };

    /** Metafunction to query a type is an iterate domain.
     */
    template <typename T, typename U, typename L, uint_t C>
    struct is_iterate_domain<iterate_domain_remapper<T, U, L, C>> : std::true_type {};
} // namespace gridtools
