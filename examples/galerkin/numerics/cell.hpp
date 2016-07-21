#pragma once

#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Shards_BasicTopologies.hpp"

namespace gdl{


    template <ushort_t order, enumtype::Shape shape>
    struct shape_select;

    template <ushort_t order>
    struct shape_select<order, enumtype::Hexa>
    {
        using type=shards::Hexahedron<>;
    };

    template <ushort_t order>
    struct shape_select<order, enumtype::Tetra>
    {
        using type=shards::Tetrahedron<>;
    };

    template <ushort_t order>
    struct shape_select<order, enumtype::Quad>
    {
        using type=shards::Quadrilateral<>;
    };

    template <ushort_t order>
    struct shape_select<order, enumtype::Tri>
    {
        using type=shards::Triangle<>;
    };


    template <ushort_t order>
    struct shape_select<order, enumtype::Line>
    {
        using type=shards::Line<>;
    };

    template <ushort_t Order, enumtype::Shape ShapeType>
    struct cell{
        static shards::CellTopology value;
        static const enumtype::Shape shape=ShapeType;
    };

    template <ushort_t Order, enumtype::Shape ShapeType>
    shards::CellTopology cell<Order, ShapeType>::value = shards::getCellTopologyData< typename shape_select<Order,ShapeType>::type >(); // cell type: hexahedron


    template <enumtype::Shape S>
    struct shape_property;

    template <>
    struct shape_property<enumtype::Hexa>{
        static const ushort_t dimension=3;
        static const ushort_t n_sub_cells=6;
        static const enumtype::Shape boundary=enumtype::Quad;

        // see definitions in Shards_BasicTopologies.hpp
        template<ushort_t FaceOrd>
        struct tangent_u
        {
            static const ushort_t value=69;
        };

        template<ushort_t FaceOrd>
        struct tangent_v{
            static const ushort_t value=69;
        };


        template<ushort_t FaceOrd>
        struct normal;

        template<ushort_t FaceOrd>
        struct opposite{
            static const ushort_t value= FaceOrd%2 ? /*odd*/ FaceOrd+1 : /*even*/ FaceOrd-1;
            //(God bless the Shards library)
            //unfortunately the convention changes for quadrilaterals
        };
    };

    template <>
    struct shape_property<enumtype::Tetra>{
        static const ushort_t dimension=3;
        static const ushort_t n_sub_cells=4;
        static const enumtype::Shape boundary=enumtype::Tri;
    };

    template <>
    struct shape_property<enumtype::Quad>{
        static const ushort_t dimension=2;
        static const ushort_t n_sub_cells=4;
        static const enumtype::Shape boundary=enumtype::Line;
    };

    template <>
    struct shape_property<enumtype::Tri>{
        static const ushort_t dimension=2;
        static const ushort_t n_sub_cells=3;
        static const enumtype::Shape boundary=enumtype::Line;
    };

    template <>
    struct shape_property<enumtype::Line>{
        static const ushort_t dimension=1;
        static const ushort_t n_sub_cells=2;
        static const enumtype::Shape boundary=enumtype::Point;
    };

    template <>
    struct shape_property<enumtype::Point>{
        static const ushort_t dimension=0;
    };

    // const ushort_t shape_property<enumtype::Hexa>::dimension;

    // const ushort_t shape_property<enumtype::Tetra>::dimension;

    // const ushort_t shape_property<enumtype::Quad>::dimension;

    // const ushort_t shape_property<enumtype::Tri>::dimension;

} //namespace gridtools
