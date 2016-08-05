#include "compute_assembly.hpp"
namespace legendre{

    compute_assembly::compute_assembly(legendre_advection& rep_) :
        m_rep(rep_),
        m_domain(boost::fusion::make_vector(  &m_rep.as_base().grid()
                                              , &m_rep.assembler().jac()
                                              , &m_rep.assembler().fe_backend().cub_weights()
                                              , &m_rep.assembler().jac_det()
                                              , &m_rep.assembler().jac_inv()
                                              , &m_rep.assembler().fe_backend().grad()
                                              , &m_rep.bd_assembler().bd_jac()
                                              , &m_rep.bd_assembler().normals()
                                              , &m_rep.bd_assembler().bd_measure()
                                              , &m_rep.bd_assembler().bd_backend().bd_cub_weights()
                                              , &m_rep.bd_assembler().bd_backend().ref_normals()
                                              , &m_rep.bd_mass()
                                              , &m_rep.bd_mass_uv()
                                              , &m_rep.bd_assembler().bd_backend().val()
                                              , &m_rep.bd_assembler().bd_backend().grad()
                                              , &m_rep.bd_assembler().flux()
                                              , &m_rep.u()
                                              , &m_rep.result()
                                              , &m_rep.mass()
                                              , &m_rep.advection()
                                              , &m_rep.beta_phys()
                                              , &m_rep.beta_interp()
                                              , &m_rep.fe().val()
                                              , &m_rep.fe().grad()
                                              , &m_rep.bd_beta_n()
                                              , &m_rep.normals()
                     ))
    {}

    // std::shared_ptr<gt::computation>
    void compute_assembly::assemble(){

        auto compute_assembly=gt::make_computation< BACKEND >(
            m_domain, m_rep.coords()
            , gt::make_multistage(
                execute<forward>()

                // boundary fluxes

                , gt::make_stage<functors::bd_mass<as_bd::boundary_t, as_bd::bd_cub> >(p_bd_measure(), p_bd_weights(), p_bd_phi(), p_bd_phi(), p_bd_mass_uu())
                , gt::make_stage<functors::bd_mass_uv<as_bd::boundary_t, as_bd::bd_cub> >(p_bd_measure(), p_bd_weights(), p_bd_phi(), p_bd_phi(), p_bd_mass_uv())

                // Internal element

                //compute the Jacobian matrix
                , gt::make_stage<functors::update_jac<as::geometry_t, Hexa> >(p_grid_points(), p_dphi(), p_jac())
                // compute the measure (det(J))
                , gt::make_stage<functors::det<geo_t> >(p_jac(), p_jac_det())

                // interpolate beta
                , gt::make_stage< functors::transform_vec >( p_jac_det(), p_weights(), p_phi_discr(), p_beta_phys(), p_beta_interp() )

                // compute the mass matrix
                , gt::make_stage< functors::mass >(p_jac_det(), p_weights(), p_phi_discr(), p_phi_discr(), p_mass()) //mass
                // compute the advection matrix
                , gt::make_stage<functors::advection< geo_t, cub > >(p_jac_det(), p_jac_inv(), p_weights(), p_beta_interp(), p_dphi_discr(), p_phi_discr(), p_advection()) //advection

                // computing flux/discretize

                // initialize result=0
                //, gt::make_stage< functors::assign<4,int,0> >( p_result() )
                // compute the face normals: \f$ n=J*(\hat n) \f$
                , gt::make_stage<functors::compute_face_normals<as_bd::boundary_t> >(p_bd_jac(), p_ref_normals(), p_normals())
                // interpolate the normals \f$ n=\sum_i <n,\phi_i>\phi_i(x) \f$
                , gt::make_stage<functors::bd_integrate<as_bd::boundary_t> >(p_bd_phi(), p_bd_measure(), p_bd_weights(), p_normals(), p_int_normals())
                // project beta on the normal direction on the boundary \f$ \beta_n = M<\beta,n> \f$
                // note that beta is defined in the current space, so we take the scalar product with
                // the normals on the current configuration, i.e. \f$F\hat n\f$
                , gt::make_stage<functors::project_on_boundary>(p_beta_interp(), p_int_normals(), p_bd_mass_uu(), p_beta_n())
                //, gt::make_stage<functors::upwind>(p_u(), p_beta_n(), p_bd_mass_uu(), p_bd_mass_uv(),  p_result())

                // // Optional: assemble the result vector by summing the values on the element boundaries
                // // , gt::make_stage< functors::assemble<geo_t> >( p_result(), p_result() )
                // // for visualization: the result is replicated
                // // , gt::make_stage< functors::uniform<geo_t> >( p_result(), p_result() )
                // // , gt::make_stage< time_advance >(p_u(), p_result())
                ));
        compute_assembly->ready();
        compute_assembly->steady();
        compute_assembly->run();
        compute_assembly->finalize();
#ifndef CUDA_EXAMPLE
        std::cout << compute_assembly->print_meter() << std::endl;
#endif
    // return compute_assembly;
}
}//namespace legendre
