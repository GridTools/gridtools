#pragma once
#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG

// #ifdef __CUDACC__
// #include <boost/shared_ptr.hpp>
// #endif

#include "mesh.hpp"

namespace legendre{

struct legendre_advection {

    static const uint_t edge_nodes=tensor_product_element<1,order_geom>::n_points::value;

    legendre_advection(uint_t d1, uint_t d2, uint_t d3);
    void run();
    void eval();
    void finalize();

    discr_t const& fe() const { return m_fe;}

    //non const version
    discr_t& fe() { return m_fe;}

    geo_t const& geo() const { return m_geo;}

    //non const version
    geo_t & geo() { return m_geo;}

    bd_geo_cub_t const& bd_cub_geo() const { return m_bd_cub_geo;}

    //non const version
    bd_geo_cub_t & bd_cub_geo()  { return m_bd_cub_geo;}

    bd_geo_t const& bd_geo() const { return m_bd_geo;}//face ordinals

    //non const version
    bd_geo_t & bd_geo()  { return m_bd_geo;}//face ordinals

    bd_discr_cub_t const& bd_cub_discr() const { return m_bd_cub_discr;}

    //non const version
    bd_discr_cub_t & bd_cub_discr()  { return m_bd_cub_discr;}

    bd_discr_t const& bd_discr() const { return m_bd_discr;}//face ordinals

    //non const version
    bd_discr_t & bd_discr()  { return m_bd_discr;}//face ordinals

    gt::array<uint_t, 3> const& dims() const { return m_dims;}

    //non const version
    gt::array<uint_t, 3> & dims()  { return m_dims;}

    mesh const& get_mesh() const { return m_mesh;}

    //non const version
    mesh & get_mesh()  { return m_mesh;}

    as const& assembler() const { return m_assembler;}

    //non const version
    as& assembler() { return m_assembler;}

    as_bd const& bd_assembler() const { return m_bd_assembler;}

    //non const version
    as_bd & bd_assembler()  { return m_bd_assembler;}

    meta_local_t const& meta_local() const { return m_meta_local;}

    //non const version
    meta_local_t & meta_local()  { return m_meta_local;}

    matrix_storage_info_t const& meta() const { return m_meta;}

    //non const version
    matrix_storage_info_t & meta()  { return m_meta;}

    matrix_type const& advection() const { return m_advection;}

    //non const version
    matrix_type & advection()  { return m_advection;}

    matrix_type const& mass() const { return m_mass;}

    //non const version
    matrix_type & mass()  { return m_mass;}

    bd_matrix_storage_info_t const& bd_meta() const { return m_bd_meta;}

    //non const version
    bd_matrix_storage_info_t & bd_meta()  { return m_bd_meta;}

    bd_matrix_type const& bd_mass() const { return m_bd_mass;}

    //non const version
    bd_matrix_type & bd_mass() { return m_bd_mass;}

    scalar_storage_info_t const& scalar_meta() const { return m_scalar_meta;}

    //non const version
    scalar_storage_info_t & scalar_meta()  { return m_scalar_meta;}

    vector_storage_info_t const& vec_meta() const { return m_vec_meta;}

    //non const version
    vector_storage_info_t & vec_meta()  { return m_vec_meta;}

    bd_scalar_storage_info_t const& bd_scalar_meta() const { return m_bd_scalar_meta;}

    //non const version
    bd_scalar_storage_info_t & bd_scalar_meta()  { return m_bd_scalar_meta;}

    bd_vector_storage_info_t const& bd_vector_meta() const { return m_bd_vector_meta;}

    //non const version
    bd_vector_storage_info_t & bd_vector_meta()  { return m_bd_vector_meta;}

    scalar_type const& result() const { return m_result;}//new solution

    //non const version
    scalar_type & result() { return m_result;}//new solution

    bd_scalar_type const& bd_beta_n() const { return m_bd_beta_n;}

    //non const version
    bd_scalar_type& bd_beta_n() { return m_bd_beta_n;}

    bd_vector_type const& normals() const { return m_normals;}

    //non const version
    bd_vector_type & normals()  { return m_normals;}

    bd_matrix_type const& bd_mass_uv() const { return m_bd_mass_uv;}

    //non const version
    bd_matrix_type& bd_mass_uv() { return m_bd_mass_uv;}

    scalar_type const& rhs() const { return m_rhs;}//zero rhs

    //non const version
    scalar_type & rhs()  { return m_rhs;}//zero rhs

    scalar_type const& u() const { return m_u;}//initial solution

    //non const version
    scalar_type & u()  { return m_u;}//initial solution

    physical_vec_storage_info_t const& physical_vec_info() const { return m_physical_vec_info;}

    //non const version
    physical_vec_storage_info_t & physical_vec_info()  { return m_physical_vec_info;}

    vector_type const& beta_interp() const { return m_beta_interp;}

    //non const version
    vector_type & beta_interp() { return m_beta_interp;}

    physical_vec_storage_type const& beta_phys() const { return m_beta_phys;}

    //non const version
    physical_vec_storage_type & beta_phys()  { return m_beta_phys;}

    gt::grid<axis> const& coords() const { return m_coords;}

    //non const version
    gt::grid<axis> & coords()  { return m_coords;}

    physical_scalar_storage_type const& result_interpolated() const { return m_result_interpolated;}

    //non const version
    physical_scalar_storage_type & result_interpolated()  { return m_result_interpolated;}

    as_base_t::grid_type const& grid() const { return m_as_base.get_grid();}

    //non const version
    as_base_t::grid_type & grid() { return m_as_base.grid();}

    as_base_t const& as_base() const { return m_as_base;}

    //non const version
    as_base_t& as_base() { return m_as_base;}
    // as_base_t::grid_type & grid() {return m_as_base.grid();}


private:
    discr_t m_fe;
    geo_t m_geo;
    bd_geo_cub_t m_bd_cub_geo;
    bd_geo_t m_bd_geo;//face ordinals
    bd_discr_cub_t m_bd_cub_discr;
    bd_discr_t m_bd_discr;//face ordinals
    gt::array<uint_t, 3> m_dims;
    mesh m_mesh;
    as m_assembler;
    as_bd m_bd_assembler;
    meta_local_t m_meta_local;
    matrix_storage_info_t m_meta;
    matrix_type m_advection;
    matrix_type m_mass;
    bd_matrix_storage_info_t m_bd_meta;
    bd_matrix_type m_bd_mass;
    scalar_storage_info_t m_scalar_meta;
    vector_storage_info_t m_vec_meta;
    bd_scalar_storage_info_t m_bd_scalar_meta;
    bd_vector_storage_info_t m_bd_vector_meta;
    scalar_type m_result;//new solution
    // scalar_type unified_result_(scalar_meta_, 0., "unified result");//new solution
    bd_scalar_type m_bd_beta_n;
    bd_vector_type m_normals;
    bd_matrix_type m_bd_mass_uv;
    scalar_type m_rhs;//zero rhs
    scalar_type m_u;//initial solution
    physical_vec_storage_info_t m_physical_vec_info;
    vector_type m_beta_interp;
    physical_vec_storage_type m_beta_phys;
    physical_scalar_storage_info_t m_physical_scalar_storage_info;
    physical_scalar_storage_type m_result_interpolated;

    gt::array<uint_t, 5> m_dx;
    gt::array<uint_t, 5> m_dy;
    gt::grid<axis> m_coords;

    as_base_t m_as_base;

    std::shared_ptr< gt::computation > m_iteration, m_evaluation;

};
}
