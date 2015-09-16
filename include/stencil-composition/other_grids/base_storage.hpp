#pragma once
#include <storage/wrap_pointer.hpp>

namespace gridtools {

    template <typename PointerType, typename LayoutMap, typename LocationType>
    struct base_storage {
        using value_type = typename PointerType::pointee_t;
        using layout = LayoutMap;
        using location_type = LocationType;

        uint_t m_size;
        value_type* m_ptr;

        base_storage(uint_t size)
            : m_size(size)
        {
            m_ptr = new value_type[m_size];
        }

        value_type* min_addr() const {
            return m_ptr;
        }
    };

} // namespace gridtools
