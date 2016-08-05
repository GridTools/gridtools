#include "compute_jacobian.hpp"
namespace legendre{

    compute_jacobian::compute_jacobian(legendre_advection& rep_) :
        m_rep(rep_),
        m_domain(boost::fusion::make_vector(
                     &m_rep.as_base().grid()
                     , &m_rep.assembler().jac()
                     , &m_rep.assembler().fe_backend().cub_weights()
                     , &m_rep.assembler().jac_det()
                     , &m_rep.assembler().jac_inv()
                     , &m_rep.assembler().fe_backend().grad()
                     , &m_rep.bd_assembler().bd_jac()
                     // , &m_rep.bd_assembler().normals()
                     , &m_rep.bd_assembler().bd_measure()
                     , &m_rep.bd_assembler().bd_backend().bd_cub_weights()
                     // , &m_rep.bd_assembler().bd_backend().ref_normals()
                     // , &m_rep.bd_mass()
                     // , &m_rep.bd_mass_uv()
                     // , &m_rep.bd_assembler().bd_backend().val()
                     , &m_rep.bd_assembler().bd_backend().grad()
                     ))
                     , m_coords({0u, 0u, 0u, m_rep.dims()[0]-1, m_rep.dims()[0]},
                                {0u, 0u, 0u, m_rep.dims()[1]-1, m_rep.dims()[1]})
    {
        m_coords.value_list[0] = 0;
        m_coords.value_list[1] = m_rep.dims()[2]-1;
    }

    // std::shared_ptr<gt::computation>
    void compute_jacobian::assemble(){

        //![placeholders]


        auto compute_jacobian=gt::make_computation< BACKEND >(
            m_domain, m_coords
            , gt::make_multistage(
                execute<forward>()
                //computes the jacobian in the boundary points of each element
                , gt::make_stage<functors::update_bd_jac<as_bd::boundary_t, Hexa> >(p_grid_points(), p_bd_dphi(), p_bd_jac())
                //computes the measure of the boundaries with codimension 1 (ok, faces)
                , gt::make_stage<functors::measure<as_bd::boundary_t, 1> >(p_bd_jac(), p_bd_measure())
                // computes the mass on the element boundaries
                // compute the Jacobian matrix
                , gt::make_stage<functors::update_jac<as::geometry_t, Hexa> >(p_grid_points(), p_dphi(), p_jac())
                // compute the measure (det(J))
                , gt::make_stage<functors::det<geo_t> >(p_jac(), p_jac_det())
                // compute the jacobian inverse
                , gt::make_stage<functors::inv<geo_t> >(p_jac(), p_jac_det(), p_jac_inv())
                ));

        // return
        compute_jacobian->ready();
        compute_jacobian->steady();
        compute_jacobian->run();
        compute_jacobian->finalize();
#ifndef CUDA_EXAMPLE
        std::cout << compute_jacobian->print_meter() << std::endl;
#endif
    }
}//namespace legendre
