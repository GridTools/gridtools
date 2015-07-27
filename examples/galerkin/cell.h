#pragma once

namespace gridtools{
    template <ushort_t Order, enumtype::Shape ShapeType>
    struct cell{
        static const shards::CellTopology value;
    };

    template <ushort_t Order, enumtype::Shape ShapeType>
    const shards::CellTopology cell<Order, ShapeType>::value = shards::getCellTopologyData< typename shape_select<Order,ShapeType>::type >(); // cell type: hexahedron
} //namespace gridtools
