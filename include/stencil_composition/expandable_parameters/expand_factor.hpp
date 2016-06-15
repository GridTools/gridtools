#pragma once

/**@file expand factor*/

namespace gridtools {
    /** @brief factor determining the length of the "chunks" in an expandable parameters list */
    template < ushort_t Tile >
    struct expand_factor {
        static const ushort_t value = Tile;
    };

    template < typename T >
    struct is_expand_factor : boost::mpl::false_ {};

    template < ushort_t Tile >
    struct is_expand_factor< expand_factor< Tile > > : boost::mpl::true_ {};
} // namespace gridtools
