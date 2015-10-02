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
#include "b_splines.hpp"

//! [includes]

namespace gridtools{

    namespace enumtype{
        //! [enums]
        enum Basis {Lagrange, RT, Nedelec, BSplines};
        enum Shape {Hexa, Quad, Line, Point};
        //! [enums]
    }

    //! [enums]
    template <ushort_t order, enumtype::Basis basis, enumtype::Shape shape>
    struct basis_select;

    template<ushort_t P>
    struct basis_select<P, enumtype::BSplines, enumtype::Hexa>{
        using type=b_spline< order<P,P,P> >;
    };

    template<ushort_t P>
    struct basis_select<P, enumtype::BSplines, enumtype::Quad>{
        using type=b_spline< order<P,P> >;
    };

    template<ushort_t P>
    struct basis_select<P, enumtype::BSplines, enumtype::Line>{
        using type=b_spline< order<P> >;
    };

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

    template <ushort_t order>
    struct shape_select<order, enumtype::Line>
    {
        using type=shards::Line<>;
    };

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

    // // reference normal vectors for the hexahedron
    // // see definitions in Shards_BasicTopologies.hpp
    // template <>
    // struct shape_property<enumtype::Hexa>::normal<21>{
    //     static const constexpr array<float_type, 3> value{0, 0, -1};
    // };

    // template <>
    // struct shape_property<enumtype::Hexa>::normal<22>{
    //     static const  constexpr array<float_type, 3> value{0,0,1};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::normal<22>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::normal<23>{
    //     static const  constexpr array<float_type, 3> value{-1,0,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::normal<23>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::normal<24>{
    //     static const  constexpr array<float_type, 3> value{1,0,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::normal<24>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::normal<25>{
    //     static const  constexpr array<float_type, 3> value{0,-1,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::normal<25>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::normal<26>{
    //     static const  constexpr array<float_type, 3> value{0,1,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::normal<26>::value;

    // // template<enumtype::Shape S, ushort_t U>
    // // const constexpr array<float_type, 3>  shape_property<S>::template normal<U>::value;



    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_u<21>{
    //     static const  constexpr array<float_type, 3> value{-1,0,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_u<21>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_v<21>{
    //     static const  constexpr array<float_type, 3> value{0,-1,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_v<21>::value;


    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_u<22>{
    //     static const  constexpr array<float_type, 3> value{1,0,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_u<22>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_v<22>{
    //     static const  constexpr array<float_type, 3> value{0,-1,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_v<22>::value;


    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_u<23>{
    //     static const  constexpr array<float_type, 3> value{0,0,1};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_u<23>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_v<23>{
    //     static const  constexpr array<float_type, 3> value{0,1,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_v<23>::value;


    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_u<24>{
    //     static const  constexpr array<float_type, 3> value{0,0,1};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_u<24>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_v<24>{
    //     static const  constexpr array<float_type, 3> value{0,-1,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_v<24>::value;


    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_u<25>{
    //     static const  constexpr array<float_type, 3> value{0,1,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_u<25>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_v<25>{
    //     static const  constexpr array<float_type, 3> value{-1,0,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_v<25>::value;


    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_u<26>{
    //     static const  constexpr array<float_type, 3> value{0,1,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_u<26>::value;

    // template <>
    // struct shape_property<enumtype::Hexa>::tangent_v<26>{
    //     static const  constexpr array<float_type, 3> value{1,0,0};
    // };
    // const constexpr array<float_type, 3>  shape_property<enumtype::Hexa>::tangent_v<26>::value;


    template <>
    struct shape_property<enumtype::Quad>{
        static const ushort_t dimension=2;
        static const ushort_t n_sub_cells=4;
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


    const ushort_t shape_property<enumtype::Hexa>::dimension;

    const ushort_t shape_property<enumtype::Quad>::dimension;


    template <typename FE>
    struct boundary_shape;

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Hexa> > : public FE<Order, BasisType, enumtype::Quad>{};

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Quad> > : public FE<Order, BasisType, enumtype::Line>{};

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Line> > : public FE<Order, BasisType, enumtype::Point>{};

}//namespace gridtools
