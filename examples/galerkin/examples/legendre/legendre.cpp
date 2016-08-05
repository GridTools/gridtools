
/**
\file
*/
#include "legendre.hpp"
/**
   @brief flux F(u)

   in the equation \f$ \frac{\partial u}{\partial t}=F(u) \f$
*/

using namespace gdl;
namespace legendre{

struct flux {
    template<typename Arg>
    GT_FUNCTION
    constexpr auto operator()(Arg const& arg_) -> decltype((Arg()+Arg())/2.){
        return (arg_+arg_)/2.;
    }
};


legendre_advection::legendre_advection(uint_t d1, uint_t d2, uint_t d3):
    m_fe()
    , m_geo()
    , m_bd_cub_geo()
    , m_bd_geo(m_bd_cub_geo, 0, 1, 2, 3, 4, 5)
    , m_bd_cub_discr()
    , m_bd_discr(m_bd_cub_discr, 0, 1, 2, 3, 4, 5)
    , m_dims{d1,d2,d3}
    , m_mesh(d1,d2,d3)
    , m_assembler(m_geo,d1,d2,d3)
    , m_bd_assembler(m_bd_geo,d1,d2,d3)
    , m_meta_local(edge_nodes, edge_nodes, edge_nodes)
    , m_meta(d1,d2,d3,discr_map::basis_cardinality(),discr_map::basis_cardinality())
    , m_advection(m_meta, 0., "advection")
    , m_mass(m_meta, 0., "mass")
    , m_bd_meta(d1,d2,d3,discr_map::basis_cardinality(),discr_map::basis_cardinality(), 6/*faces*/)
    , m_bd_mass(m_bd_meta, 0., "bd mass")
    , m_scalar_meta(d1,d2,d3,discr_map::basis_cardinality())
    , m_vec_meta(d1,d2,d3,discr_map::basis_cardinality(), 3)
    , m_bd_scalar_meta(d1,d2,d3, discr_map::basis_cardinality(), bd_discr_t::s_num_boundaries )
    , m_bd_vector_meta(d1,d2,d3, discr_map::basis_cardinality(), 3, bd_discr_t::s_num_boundaries )
    , m_result(m_scalar_meta, 0., "result")
    , m_bd_beta_n(m_bd_scalar_meta, 0., "beta*n")
    , m_normals(m_bd_vector_meta, 0., "normals")
    , m_bd_mass_uv(m_bd_meta, 0., "mass uv")
    , m_rhs(m_scalar_meta, 0., "rhs")
    , m_u(m_scalar_meta, 0., "u")
    , m_physical_vec_info(d1,d2,d3,cub::numCubPoints(),3)
    , m_beta_interp(m_vec_meta, 0., "beta")
    , m_beta_phys(m_physical_vec_info, "beta interp")
    , m_dx{1, 1, 1, d1-2, d1}
    , m_dy{1, 1, 1, d2-2, d2}
    , m_coords(m_dx,m_dy)
    , m_as_base(d1, d2, d3)
    , m_physical_scalar_storage_info(d1, d2, d3 ,cub::numCubPoints())
    , m_result_interpolated(m_physical_scalar_storage_info, 0., "interpolated result")
    {
        m_fe.compute(Intrepid::OPERATOR_VALUE);
        m_fe.compute(Intrepid::OPERATOR_GRAD);
        m_geo.compute(Intrepid::OPERATOR_GRAD);

        m_bd_geo.compute(Intrepid::OPERATOR_VALUE);
        m_bd_geo.compute(Intrepid::OPERATOR_GRAD);
        m_bd_discr.compute(Intrepid::OPERATOR_VALUE);

        m_coords.value_list[0] = 1;
        m_coords.value_list[1] = d3-2;

        //mesh construction
        for (uint_t i=0; i< m_dims[0]; i++)
            for (uint_t j=0; j< m_dims[1]; j++)
                for (uint_t k=0; k< m_dims[2]; k++)
                    for (uint_t point=0; point<geo_map::basis_cardinality(); point++)
                    {
                        m_as_base.grid()( i,  j,  k,  point,  0)= (i + (1+m_geo.grid()(point, 0, 0))/2.)/m_dims[0];
                        m_as_base.grid()( i,  j,  k,  point,  1)= (j + (1+m_geo.grid()(point, 1, 0))/2.)/m_dims[1];
                        m_as_base.grid()( i,  j,  k,  point,  2)= (k + (1+m_geo.grid()(point, 2, 0))/2.)/m_dims[2];
                    }

    // initialization
    for (uint_t i=0; i<m_dims[0]; i++)
        for (uint_t j=0; j<m_dims[1]; j++)
            for (uint_t k=0; k<m_dims[2]; k++)
            {
                for (uint_t point=0; point<cub::numCubPoints(); point++)
                {
                    auto norm=std::sqrt(
                        gt::gt_pow<2>::apply(m_as_base.grid()( i,  j,  k,  0,  0) + m_assembler.fe_backend().cub_points()(point, 0, 0))
                        +
                        gt::gt_pow<2>::apply(m_as_base.grid()( i,  j,  k,  0,  1) + m_assembler.fe_backend().cub_points()(point, 1, 0)));

                    if(norm)
                    {
                        m_beta_phys(i,j,k,point,0)=-(m_as_base.grid()( i,  j,  k,  0,  1) + m_assembler.fe_backend().cub_points()(point, 0, 0))/norm;
                        m_beta_phys(i,j,k,point,1)=(m_as_base.grid()( i,  j,  k,  0,  0) + m_assembler.fe_backend().cub_points()(point, 1, 0))/norm;
                    }
                    else{
                        m_beta_phys(i,j,k,point,0)=0.;
                        m_beta_phys(i,j,k,point,1)=0.;
                    }
                    m_beta_phys(i,j,k,point,2)=0.;
                }
                for (uint_t point=0; point<discr_map::basis_cardinality(); point++)
                {
                    // if(i+j+k>4)
                    if( i==2 && j>3 && j<6 )
                        m_u(i,j,k,point)=1.;//point;
                }
            }
    }


    void legendre_advection::run(){
    struct it{
        typedef  gt::arg<0, bd_matrix_type> p_bd_mass_uu;
        typedef  gt::arg<1, bd_matrix_type> p_bd_mass_uv;
        typedef  gt::arg<2, scalar_type> p_u;
        typedef  gt::arg<3, scalar_type> p_result;
        typedef  gt::arg<4, matrix_type> p_mass;
        typedef  gt::arg<5, matrix_type> p_advection;
        typedef  gt::arg<6, bd_scalar_type> p_beta_n;
        typedef  gt::arg<7, scalar_type> p_rhs;
    };

    typedef typename boost::mpl::vector< it::p_bd_mass_uu
                                         , it::p_bd_mass_uv
                                         , it::p_u
                                         , it::p_result
                                         , it::p_mass
                                         , it::p_advection
                                         , it::p_beta_n
                                         , it::p_rhs > mpl_list_iteration;

    gt::aggregator_type<mpl_list_iteration> domain_iteration(boost::fusion::make_vector( &m_bd_mass
                                                                                 , &m_bd_mass_uv
                                                                                 , &m_u
                                                                                 , &m_result
                                                                                 , &m_mass
                                                                                 , &m_advection
                                                                                 , &m_bd_beta_n
                                                                                 , &m_rhs
                                                         ));

    m_iteration=gt::make_computation< BACKEND >(
         domain_iteration, m_coords
         , gt::make_multistage (
             execute<forward>()
             , gt::make_stage< functors::assign<4,int,0> >( it::p_result() )
             // add the advection term: result+=A*u
             , gt::make_stage< functors::matvec >( it::p_advection(), it::p_u(), it::p_result() )
             //compute the upwind flux
             //i.e.:
             //if <beta,n> > 0
             // result= <beta,n> * [(u+ * v+) - (u+ * v-)]
             //if beta*n<0
             // result= <beta,n> * [(u- * v-) - (u- * v+)]
             // where + means "this element" and - "the neighbour"
             , gt::make_stage< functors::upwind>(it::p_u(), it::p_beta_n(), it::p_bd_mass_uu(), it::p_bd_mass_uv(),  it::p_result())
             // add the advection term (for time dependent problem): result+=A*u
             //, gt::make_stage< functors::matvec>( it::p_u(), it::p_mass(), it::p_result() )
             //, gt::make_stage<residual>(it::p_rhs(), it::p_result(), it::p_u()) //updating u = u - (Ax-rhs)
             )
        );

    m_iteration->ready();
    m_iteration->steady();
    m_iteration->run();

    }

    void legendre_advection::eval(){
        struct evaluate{

            typedef  gt::arg<0, typename discr_t::basis_function_storage_t> p_phi;
            typedef  gt::arg<1, scalar_type> p_result;
            typedef  gt::arg<2, physical_scalar_storage_type> p_result_interpolated;
            typedef  gt::arg<3, typename as::storage_type >    p_jac_det;
            typedef  gt::arg<4, typename as::geometry_t::weights_storage_t >   p_weights;
        };

        typedef typename boost::mpl::vector< evaluate::p_phi, evaluate::p_result, evaluate::p_result_interpolated, evaluate::p_jac_det, evaluate::p_weights> mpl_list_interp;

        gt::aggregator_type<mpl_list_interp> domain_interp(boost::fusion::make_vector(
                                                       &m_fe.val()
                                                       ,&m_result
                                                       ,&m_result_interpolated
                                                       ,&m_assembler.jac_det()
                                                       ,&m_assembler.fe_backend().cub_weights()
                                                       ));

        m_evaluation=gt::make_computation< BACKEND >(
            domain_interp, m_coords
            , gt::make_multistage(
                execute<forward>()
                , gt::make_stage< functors::evaluate >( evaluate::p_phi(), evaluate::p_result(), evaluate::p_weights(), evaluate::p_jac_det(), evaluate::p_result_interpolated() )
                )
            );

        m_evaluation->ready();
        m_evaluation->steady();
        m_evaluation->run();

// //![computation]
//    // intrepid::test(assembler, bd_discr_, bd_mass_);
    }

    void legendre_advection::finalize(){
        if(m_iteration.get())
        {
            m_iteration->finalize();
#ifndef CUDA_EXAMPLE
            std::cout << m_iteration->print_meter() << std::endl;
#endif
        }
        if(m_evaluation.get())
            {
            m_evaluation->finalize();
#ifndef CUDA_EXAMPLE
            std::cout << m_evaluation->print_meter() << std::endl;
#endif
            }
    }
}//namespace legendre
