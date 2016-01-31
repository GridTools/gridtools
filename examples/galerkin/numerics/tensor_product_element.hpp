#pragma once

namespace gdl{
    template <int_t N>
    struct factorial{
        static const int_t value=N*factorial<N-1>::value;
    };

    template<>
    struct factorial<0>{
        static const int_t value=1;
    };

    template <int_t n, int_t m>
    struct combinations{
        static_assert(n>=m, "wrong combination");
        static const int_t value = (int_t) n==m ? 0 : factorial<n>::value/(factorial<m>::value * factorial<n-m>::value);
    };

    template<int_t dimension, int_t order>
    struct tensor_product_element{

        using dimension_t=static_ushort<dimension>;
        using order_t=static_ushort<order>;

        // boundary type with the given codimension
        template<int_t codimension>
        using  boundary_w_codim = tensor_product_element<dimension-codimension, order>;

        template<int_t sub_dimension>
        using  boundary_w_dim = tensor_product_element<sub_dimension, order>;

        // number of boundaries with given codimension
        // evaluates to 0 in case sub_dimension >= dimension
        template<int_t sub_dimension>
        using n_boundary_w_dim= static_int<sub_dimension>=dimension ? 0 : (gt::gt_pow<dimension-sub_dimension>::apply(2))*combinations<dimension, sub_dimension >::value>;

        // number of vertices
        using n_vertices = static_int<gt::gt_pow<dimension>::apply(2)>;

        // total number of points
        using n_points = static_int<gt::gt_pow<dimension>::apply(2+(order-1))>;

        // number of "internal" points
        // or on boundary with dimension sub_dim
        template<int_t sub_dimension=dimension>
        using n_internal_points = static_int< gt::gt_pow<dimension>::apply(2+(order-3)) +  gt::gt_pow<sub_dimension>::apply(2+(order-3)) * n_boundary_w_dim<sub_dimension>::value >;

        // number of boundary points
        template<int_t sub_dimension=dimension>
        using n_boundary_points=static_int<n_points::value - n_internal_points<sub_dimension>::value >;

    };

    static_assert(tensor_product_element<3,1>::n_vertices::value==8, "error");
    static_assert(tensor_product_element<3,1>::n_points::value==8, "error");
    static_assert(tensor_product_element<3,1>::n_boundary_w_dim<2>::value==6, "error");
    static_assert(tensor_product_element<3,1>::n_boundary_w_dim<1>::value==12, "error");
    static_assert(tensor_product_element<3,1>::boundary_w_codim<1>::n_points::value==4, "error");
    static_assert(tensor_product_element<3,1>::n_internal_points<1>::value==0, "error");
    static_assert(tensor_product_element<3,1>::n_boundary_points<1>::value==8, "error");

    static_assert(tensor_product_element<3,2>::n_vertices::value==8, "error");
    static_assert(tensor_product_element<3,2>::n_points::value==27, "error");
    static_assert(tensor_product_element<3,2>::n_internal_points<>::value==1, "error");
    static_assert(tensor_product_element<3,2>::n_boundary_points<>::value==26, "error");
    static_assert(tensor_product_element<3,2>::n_boundary_points<2>::value==20, "error");
    static_assert(tensor_product_element<3,2>::n_boundary_points<1>::value==14, "error");
} // namespace gridtools
