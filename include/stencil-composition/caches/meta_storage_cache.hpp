#pragma once
#include "common/storage_info_interface.hpp"

#include "../../common/generic_metafunctions/unzip.hpp"

namespace gridtools {

    template < typename Layout, uint_t... Dims >
    struct meta_storage_cache {

        typedef storage_info_interface< 0, Layout > meta_storage_t;
        typedef Layout layout_t;

      public:
        static constexpr meta_storage_t m_value { Dims... };

        GT_FUNCTION
        static constexpr uint_t size() { return m_value.size(); }

        template < ushort_t Id >
        GT_FUNCTION static constexpr int_t stride() {
            return m_value.template stride< Id >();
        }

        template < ushort_t Id >
        GT_FUNCTION static constexpr int_t dim() {
            return m_value.template dim< Id >();
        }

    };
} // namespace gridtools
