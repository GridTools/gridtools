#pragma once

namespace gridtools{
    template <ushort_t Order, enumtype::Shape ShapeType>
    struct cell{
        static shards::CellTopology value;
        static const enumtype::Shape shape=ShapeType;
    };

    template <ushort_t Order, enumtype::Shape ShapeType>
    shards::CellTopology cell<Order, ShapeType>::value = shards::getCellTopologyData< typename shape_select<Order,ShapeType>::type >(); // cell type: hexahedron
} //namespace gridtools
