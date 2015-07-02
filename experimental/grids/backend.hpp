#pragma once

#include "base_storage.hpp"
#include "location_type.hpp"


namespace gridtools {
    struct _backend {
    private:
        template <typename LocationType, typename X>
        struct _storage_type;

        template <typename X>
        struct _storage_type<location_type<0>, X> {
            using type = base_storage<wrap_pointer<double>, layout_map<0,1,2>, location_type<0> >;
        };

        template <typename X>
        struct _storage_type<location_type<1>, X> {
            using type = base_storage<wrap_pointer<double>, layout_map<0,1,2>, location_type<1> >;
        };

    public:
        template <typename LocationType>
        using storage_type = typename _storage_type<LocationType, void>::type;
    };
} // namespace gridtools
