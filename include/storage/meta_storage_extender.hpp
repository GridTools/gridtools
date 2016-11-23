/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "meta_storage.hpp"

#ifdef CXX11_ENABLED
namespace gridtools {

    /**
     * @brief The extend_aux_param struct
     * it extends the declaration of a template parameter used by metastorage by NExtraDim dimensions
     */
    // specialization for parameters that are dimension independent, the metafunction has no impact
    template < ushort_t NExtraDim, typename T >
    struct extend_aux_param {
        typedef T type;
    };

    // specialization for a halo template parameter
    template < ushort_t NExtraDim, uint_t... Args >
    struct extend_aux_param< NExtraDim, halo< Args... > > {
        typedef typename repeat_template_c< 0, NExtraDim, halo, Args... >::type type;
    };

    template < typename T, ushort_t NExtraDim >
    struct meta_storage_extender_impl;

    template < template < typename... > class Base, typename First, ushort_t NExtraDim, typename... TmpParam >
    struct meta_storage_extender_impl< Base< First, TmpParam... >, NExtraDim > {
        GRIDTOOLS_STATIC_ASSERT((!is_meta_storage_tmp< Base< First, TmpParam... > >::value),
            "Meta storage extender is not supported for tmp meta storages");

        typedef Base< typename meta_storage_extender_impl< First, NExtraDim >::type,
            typename extend_aux_param< NExtraDim, TmpParam >::type... >
            type;
    };

    template < short_t Val, short_t NExtraDim >
    struct inc_ {
        static const short_t value = Val + NExtraDim;
    };

    template < typename Index, typename Layout, bool TmpFlag, ushort_t NExtraDim >
    struct meta_storage_extender_impl< meta_storage_base< Index, Layout, TmpFlag >, NExtraDim > {
        typedef meta_storage_base< Index, typename meta_storage_extender_impl< Layout, NExtraDim >::type, TmpFlag >
            type;
    };

    template < ushort_t NExtraDim, short_t... Args >
    struct meta_storage_extender_impl< layout_map< Args... >, NExtraDim > {

        template < typename T, short_t... InitialInts >
        struct build_ext_layout;

        // build an extended layout
        template < short_t... Indices, short_t... InitialIndices >
        struct build_ext_layout< gt_integer_sequence< short_t, Indices... >, InitialIndices... > {
            typedef layout_map< InitialIndices..., Indices... > type;
        };

        using seq = typename make_gt_integer_sequence< short_t, NExtraDim >::type;

        typedef typename build_ext_layout< seq, inc_< Args, NExtraDim >::value... >::type type;
    };

    template < ushort_t Index, typename Layout, bool IsTemporary, ushort_t NExtraDim >
    struct meta_storage_extender_impl< meta_storage_base< static_int< Index >, Layout, IsTemporary >, NExtraDim > {
        typedef meta_storage_base< static_int< Index >,
            typename meta_storage_extender_impl< Layout, NExtraDim >::type,
            IsTemporary >
            type;
    };

    /**
     * @brief The meta_storage_extender struct
     * helper that extends a metastorage by certain number of dimensions. Lengths of the extra dimensions are passed by
     * arguments. Values of halos of extra dims are set to null, and the layout of the new meta storage is such that the
     * newly added dimensions have the largest stride.
     */
    struct meta_storage_extender {
        template < typename MetaStorage >
        typename meta_storage_extender_impl< MetaStorage, 1 >::type operator()(
            const MetaStorage other, uint_t extradim_length) {
            GRIDTOOLS_STATIC_ASSERT((is_meta_storage< MetaStorage >::value), "Use with a MetaStorage type only");
            typedef typename meta_storage_extender_impl< MetaStorage, 1 >::type type;

            const array< uint_t, MetaStorage::space_dimensions > dims = other.unaligned_dims();
            auto ext_dim = dims.append_dim(extradim_length);
            return type(ext_dim);
        }
    };
}
#endif
