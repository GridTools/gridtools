#pragma once

//this MUST be included before any boost include
#define FUSION_MAX_VECTOR_SIZE 30
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#define BACKEND_BLOCK

#define N_FACES 6
#define N_DOFS_F 8
#define N_DOFS_F_BD 4
#define N_DOFS 56
#define N_CUB 27
#define N_CUB_BD 9

#include <common/layout_map_metafunctions.hpp>

//! [assembly]
#include "../../numerics/bd_assembly.hpp"
//! [assembly]
#include "../../numerics/tensor_product_element.hpp"
#include "../../functors/matvec.hpp"
#include "../../functors/interpolate.hpp"

namespace legendre{
    using namespace gdl;
    using namespace gdl::enumtype;

    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<5> >;
    using bd_matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<6> >; //last dimension is tha face
    using matrix_type=storage_t< matrix_storage_info_t >;
    using bd_matrix_type=storage_t< bd_matrix_storage_info_t >;
    static const ushort_t order_geom=1;
    static const ushort_t order_discr=5;

    using geo_map=reference_element<order_geom, Lagrange, Hexa>;
    using discr_map=reference_element<order_discr, Legendre, Hexa>;
    //integrate exactly polyunomials of degree (discr_map::order*geo_map::order)
    using cub=cubature<4, geo_map::shape()>;//overintegrating: few basis func are 0 on all quad points otherwise...
    using discr_t = intrepid::discretization<discr_map, cub>;
    using geo_t = intrepid::geometry<geo_map, cub>;
    //boundary
    using bd_geo_cub_t = intrepid::boundary_cub<geo_map, cub::cubDegree>;
    using bd_discr_cub_t = intrepid::boundary_cub<discr_map, cub::cubDegree>;
    using bd_geo_t = intrepid::boundary_discr<bd_geo_cub_t>;
    using bd_discr_t = intrepid::boundary_discr<bd_discr_cub_t>;

    using as=assembly<geo_t>;
    using as_bd=bd_assembly<bd_geo_t>;
    using scalar_storage_info_t=storage_info< __COUNTER__, layout_tt<4>>;//TODO change: iterate on faces
    typedef BACKEND::storage_info<0, gridtools::layout_map<0,1,2> > meta_local_t;
    using vector_storage_info_t=storage_info< __COUNTER__, layout_tt<5>>;//TODO change: iterate on faces
    using bd_scalar_storage_info_t=storage_info< __COUNTER__, layout_tt<5>>;//TODO change: iterate on faces
    using bd_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<6>>;//TODO change: iterate on
    using scalar_type=storage_t< scalar_storage_info_t >;
    using vector_type=storage_t< vector_storage_info_t >;
    using bd_scalar_type=storage_t< bd_scalar_storage_info_t >;
    using bd_vector_type=storage_t< bd_vector_storage_info_t >;

    using physical_scalar_storage_info_t = storage_info< __COUNTER__, layout_tt<4> >;
    using physical_vec_storage_info_t =storage_info< __COUNTER__, layout_tt<5> >;
    using physical_scalar_storage_type = storage_t<physical_scalar_storage_info_t>;
    using physical_vec_storage_type = storage_t<physical_vec_storage_info_t>;

    using as_base_t=assembly_base<geo_t>;

    class mesh{
    private:
    // as_base_t m_as_base;
    public:
        mesh(uint_t d1, uint_t d2, uint_t d3);
        // as_base_t::grid_type const& get_grid() const {return m_as_base.get_grid();}
   };
}
