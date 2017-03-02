#pragma once
#include "common/storage_info_interface.hpp"

#include "../../common/generic_metafunctions/unzip.hpp"

namespace gridtools {

    template < typename Layout, uint_t... Dims >
    struct meta_storage_cache {

        typedef storage_info_interface< 0, Layout > meta_storage_t;
        typedef Layout layout_t;

      public:
        const meta_storage_t m_value;

        GT_FUNCTION
        constexpr meta_storage_cache(meta_storage_cache const &other) : m_value{other.m_value} {};

        /**NOTE: the last 2 dimensions are Component and FD by convention*/
        GT_FUNCTION
        constexpr meta_storage_cache() : m_value{Dims...} {};

        GT_FUNCTION
        constexpr meta_storage_t value() const { return m_value; }

        GT_FUNCTION
        constexpr uint_t size() const { return m_value.size(); }

        template < ushort_t Id >
        GT_FUNCTION constexpr int_t stride() const {
            return m_value.template stride< Id >();
        }

        template < ushort_t Id >
        GT_FUNCTION constexpr int_t dim() const {
            return m_value.template dim< Id >();
        }

    };
} // namespace gridtools
