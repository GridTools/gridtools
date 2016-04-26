#pragma once

namespace gridtools{
    template <ushort_t Tile>
    struct expand_factor{
        static const ushort_t value=Tile;
    };
} //namespace gridtools
