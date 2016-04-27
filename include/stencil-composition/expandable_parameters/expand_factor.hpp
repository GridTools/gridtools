#pragma once

/**@file expand factor*/

namespace gridtools{
    /** @brief factor determining the length of the "chunks" in an expandable parameters list */
    template <ushort_t Tile>
    struct expand_factor{
        static const ushort_t value=Tile;
    };
} //namespace gridtools
