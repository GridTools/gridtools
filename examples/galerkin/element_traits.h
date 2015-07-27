#pragma once
//! [includes]
#include <Intrepid_Basis.hpp>
#include <Intrepid_Types.hpp>
#include <Intrepid_FieldContainer.hpp>
#include "Intrepid_HGRAD_TET_Cn_FEM_ORTH.hpp"
#include "Intrepid_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HDIV_HEX_In_FEM.hpp"

#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Shards_BasicTopologies.hpp"
//! [includes]

namespace gridtools{

    namespace enumtype{
        //! [enums]
        enum Basis {Lagrange, RT, Nedelec};
        enum Shape {Hexa, Quad, Line, Point};
        //! [enums]
    }

    //! [enums]
    template <ushort_t order, enumtype::Basis basis, enumtype::Shape shape>
    struct basis_select;

    template<>
    struct basis_select<1, enumtype::Lagrange, enumtype::Hexa>{
        using type=Intrepid::Basis_HGRAD_HEX_C1_FEM<double, Intrepid::FieldContainer<double> >;
    };

    template<ushort_t order>
    struct basis_select<order, enumtype::Lagrange, enumtype::Hexa>{
        using type=Intrepid::Basis_HGRAD_HEX_Cn_FEM<double, Intrepid::FieldContainer<double> >;
    };

    template<ushort_t order>
    struct basis_select<order, enumtype::Lagrange, enumtype::Quad>{
        using type=Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >;
    };

    template<>
    struct basis_select<1, enumtype::Lagrange, enumtype::Quad>{
        using type=Intrepid::Basis_HGRAD_QUAD_C1_FEM<double, Intrepid::FieldContainer<double> >;
    };

    template <ushort_t order, enumtype::Shape shape>
    struct shape_select;

    template <ushort_t order>
    struct shape_select<order, enumtype::Hexa>
    {
        using type=shards::Hexahedron<>;
    };

    template <ushort_t order>
    struct shape_select<order, enumtype::Quad>
    {
        using type=shards::Quadrilateral<>;
    };

    template <enumtype::Shape S>
    struct shape_property;

    template <>
    struct shape_property<enumtype::Hexa>{
        static const ushort_t dimension=3;
    };

    template <>
    struct shape_property<enumtype::Quad>{
        static const ushort_t dimension=2;
    };

    template <>
    struct shape_property<enumtype::Line>{
        static const ushort_t dimension=1;
    };

    template <>
    struct shape_property<enumtype::Point>{
        static const ushort_t dimension=0;
    };


    template <typename FE>
    struct boundary_shape;

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Hexa> > : public FE<Order, BasisType, enumtype::Quad>{};

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Quad> > : public FE<Order, BasisType, enumtype::Line>{};

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Line> > : public FE<Order, BasisType, enumtype::Point>{};

}//namespace gridtools
