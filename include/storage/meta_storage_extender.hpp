#pragma once
#include "meta_storage.hpp"

namespace gridtools {

    template < ushort_t NExtraDim, typename T >
    struct extend_aux_param {
        typedef T type;
    };

    template < ushort_t NExtraDim, uint_t... Args >
    struct extend_aux_param< NExtraDim, halo< Args... > > {
        typedef typename repeat_template_c< 0, NExtraDim, uint_t, halo, Args... >::type type;
    };

    template < typename T, ushort_t NExtraDim >
    struct meta_storage_extender_impl;

    template < template < typename... > class Base, typename First, ushort_t NExtraDim, typename... TmpParam >
    struct meta_storage_extender_impl< Base< First, TmpParam... >, NExtraDim > {
        GRIDTOOLS_STATIC_ASSERT((!is_meta_storage_tmp< Base< First, TmpParam... > >::value),
            "Meta storage extender is not supported for tmp meta storages");

        typedef Base< typename meta_storage_extender_impl< First, NExtraDim >::type,
            typename extend_aux_param< NExtraDim, TmpParam >::type... > type;
    };

    template < short_t Val >
    struct inc_ {
        static const short_t value = Val + 1;
    };

    template < ushort_t NExtraDim, short_t... Args >
    struct meta_storage_extender_impl< layout_map< Args... >, NExtraDim > {
        typedef typename repeat_template_c< 0, NExtraDim, short_t, layout_map, inc_< Args >::value... >::type type;
    };

    template < ushort_t Index, typename Layout, bool IsTemporary, ushort_t NExtraDim >
    struct meta_storage_extender_impl< meta_storage_base< Index, Layout, IsTemporary >, NExtraDim > {
        typedef meta_storage_base< Index, typename meta_storage_extender_impl< Layout, NExtraDim >::type, IsTemporary >
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

            // TODO once available we should use unaligned dim
            auto dims = other.dims();
            auto ext_dim = dims.append_dim(extradim_length);
            return type(ext_dim);
        }
    };
}
