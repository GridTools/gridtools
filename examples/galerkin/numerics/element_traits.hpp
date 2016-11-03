/**
\file
*/

#pragma once
//! [includes]
#include <Intrepid_Basis.hpp>
#include <Intrepid_Types.hpp>
#include <Intrepid_FieldContainer.hpp>
#include "Intrepid_HGRAD_TET_COMP12_FEM.hpp"
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

#include "legendre.hpp"
#ifndef __CUDACC__ //not yet
#include "b_splines.hpp"
#endif
//! [includes]

namespace gdl{


    //! [enums]
    template <ushort_t order, enumtype::Basis basis, enumtype::Shape shape>
    struct basis_select;

#ifndef __CUDACC__ //not yet
    template<ushort_t P>
    struct basis_select<P, enumtype::BSplines, enumtype::Hexa>{
        using type=b_spline< order<P,P,P> >;
        static type instance(){return type();}
    };

    template<ushort_t P>
    struct basis_select<P, enumtype::BSplines, enumtype::Tetra>{
        using type=b_spline< order<P,P,P> >;
        static type instance(){return type();}
    };

    template<ushort_t P>
    struct basis_select<P, enumtype::BSplines, enumtype::Quad>{
        using type=b_spline< order<P,P> >;
        static type instance(){return type();}
    };

    template<ushort_t P>
    struct basis_select<P, enumtype::BSplines, enumtype::Line>{
        using type=b_spline< order<P> >;
        static type instance(){return type();}
    };
#endif

    template<ushort_t P>
    struct basis_select<P, enumtype::Legendre, enumtype::Hexa>{
        using type=legendre< 3, P >;
        static type instance(){return type();}
    };

    // template<ushort_t P>
    // struct basis_select<P, enumtype::Legendre, enumtype::Tetra>{
    //     using type=legendre< 3, P >;
    //     static type instance(){return type();}
    // };

    template<ushort_t P>
    struct basis_select<P, enumtype::Legendre, enumtype::Quad>{
        using type=legendre< 2, P >;
        static type instance(){return type();}
    };

    template<ushort_t P>
    struct basis_select<P, enumtype::Legendre, enumtype::Line>{
        using type=legendre< 1, P>;
        static type instance(){return type();}
    };

    template<>
    struct basis_select<1, enumtype::Lagrange, enumtype::Hexa>{
        using type=Intrepid::Basis_HGRAD_HEX_C1_FEM<gt::float_type, Intrepid::FieldContainer<gt::float_type> >;
        static type instance(){return type();}
    };

    template<ushort_t order>
    struct basis_select<order, enumtype::Lagrange, enumtype::Hexa>{
        using type=Intrepid::Basis_HGRAD_HEX_Cn_FEM<gt::float_type, Intrepid::FieldContainer<gt::float_type> >;
        static type instance(){return type(order, Intrepid::POINTTYPE_EQUISPACED);}
    };

    template<>
    struct basis_select<1, enumtype::Lagrange, enumtype::Tetra>{
        using type=Intrepid::Basis_HGRAD_TET_C1_FEM<gt::float_type, Intrepid::FieldContainer<gt::float_type> >;
        static type instance(){return type();}
    };

    template<ushort_t order>
    struct basis_select<order, enumtype::Lagrange, enumtype::Tetra>{
        using type=Intrepid::Basis_HGRAD_TET_Cn_FEM<gt::float_type, Intrepid::FieldContainer<gt::float_type> >;
        static type instance(){return type(order, Intrepid::POINTTYPE_EQUISPACED);}
    };

    template<ushort_t order>
    struct basis_select<order, enumtype::Lagrange, enumtype::Quad>{
        using type=Intrepid::Basis_HGRAD_QUAD_Cn_FEM<gt::float_type, Intrepid::FieldContainer<gt::float_type> >;
        static type instance(){return type(order, Intrepid::POINTTYPE_EQUISPACED);}
    };

    template<>
    struct basis_select<1, enumtype::Lagrange, enumtype::Quad>{
        using type=Intrepid::Basis_HGRAD_QUAD_C1_FEM<gt::float_type, Intrepid::FieldContainer<gt::float_type> >;
        static type instance(){return type();}
    };

    template<ushort_t order>
    struct basis_select<order, enumtype::Lagrange, enumtype::Tri>{
        using type=Intrepid::Basis_HGRAD_TRI_Cn_FEM<gt::float_type, Intrepid::FieldContainer<gt::float_type> >;
        static type instance(){return type(order, Intrepid::POINTTYPE_EQUISPACED);}
    };

    template<>
    struct basis_select<1, enumtype::Lagrange, enumtype::Tri>{
        using type=Intrepid::Basis_HGRAD_TRI_C1_FEM<gt::float_type, Intrepid::FieldContainer<gt::float_type> >;
        static type instance(){return type();}
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



    template <typename FE>
    struct boundary_shape;

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Hexa> > : public FE<Order, BasisType, enumtype::Quad>{};

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Tetra> > : public FE<Order, BasisType, enumtype::Tri>{};

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Quad> > : public FE<Order, BasisType, enumtype::Line>{};

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Tri> > : public FE<Order, BasisType, enumtype::Line>{};

    template <uint_t Order, enumtype::Basis BasisType, template<ushort_t O, enumtype::Basis E, enumtype::Shape S > class FE>
    struct boundary_shape<FE<Order, BasisType, enumtype::Line> > : public FE<Order, BasisType, enumtype::Point>{};

}//namespace gdl
