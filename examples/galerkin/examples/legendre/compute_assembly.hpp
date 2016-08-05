#pragma once
#include "legendre.hpp"

namespace legendre{
class compute_assembly{


        typedef gt::arg<0, typename as_base_t::grid_type >       p_grid_points;
        typedef gt::arg<1, typename as::jacobian_type >   p_jac;
        typedef gt::arg<2, typename as::geometry_t::weights_storage_t >   p_weights;
        typedef gt::arg<3, typename as::storage_type >    p_jac_det;
        typedef gt::arg<4, typename as::jacobian_type >   p_jac_inv;
        // typedef gt::arg<5, typename as::geometry_t::basis_function_storage_t> p_phi;
        typedef gt::arg<5, typename as::geometry_t::grad_storage_t> p_dphi;

        typedef gt::arg<6, typename as_bd::jacobian_type >       p_bd_jac;
        typedef gt::arg<7, typename as_bd::face_normals_type >                   p_normals;
        typedef gt::arg<8, typename as_bd::storage_type >        p_bd_measure;
        typedef gt::arg<9, typename as_bd::boundary_t::weights_storage_t> p_bd_weights;
        typedef gt::arg<10, typename as_bd::boundary_t::tangent_storage_t> p_ref_normals;
        typedef gt::arg<11, bd_matrix_type> p_bd_mass_uu;
        typedef gt::arg<12, bd_matrix_type> p_bd_mass_uv;
        typedef gt::arg<13, typename as_bd::boundary_t::basis_function_storage_t> p_bd_phi;
        typedef gt::arg<14, typename as_bd::boundary_t::grad_storage_t> p_bd_dphi;
        typedef gt::arg<15, typename as_bd::bd_vector_type> p_flux;

        typedef gt::arg<16, scalar_type> p_u;
        typedef gt::arg<17, scalar_type> p_result;
        typedef gt::arg<18, matrix_type> p_mass;
        typedef gt::arg<19, matrix_type> p_advection;
        typedef gt::arg<20, physical_vec_storage_type> p_beta_phys;
        typedef gt::arg<21, vector_type> p_beta_interp;
        typedef gt::arg<22, typename discr_t::basis_function_storage_t> p_phi_discr;
        typedef gt::arg<23, typename discr_t::grad_storage_t> p_dphi_discr;
        typedef gt::arg<24, bd_scalar_type> p_beta_n;
        typedef gt::arg<25, bd_vector_type> p_int_normals;

        // typedef gt::arg<26, scalar_type> p_unified_result;

        typedef typename boost::mpl::vector<p_grid_points, p_jac, p_weights, p_jac_det, p_jac_inv, // p_phi,
                                            p_dphi, p_bd_jac, p_normals, p_bd_measure, p_bd_weights, p_ref_normals, p_bd_mass_uu , p_bd_mass_uv
                                            , p_bd_phi, p_bd_dphi, p_flux, p_u, p_result, p_mass, p_advection, p_beta_phys, p_beta_interp, p_phi_discr, p_dphi_discr, p_beta_n, p_int_normals// , p_unified_result
                                            > mpl_list;

private:
    legendre_advection& m_rep;
    gt::aggregator_type<mpl_list> m_domain;

public:
    compute_assembly( legendre_advection & rep_);
    // std::shared_ptr<gt::computation>
    void assemble();
};
} //namespace legendre
