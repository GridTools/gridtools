//this MUST be included before any boost include
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS


/**
\file
*/
#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG

//! [assembly]
#include "../numerics/bd_assembly.hpp"
//! [assembly]
#include "../numerics/tensor_product_element.hpp"
#include "../functors/matvec.hpp"
#include "../tools/io.hpp"

/**
   @brief flux F(u)

   in the equation \f$ \frac{\partial u}{\partial t}=F(u) \f$
*/
struct flux {
    template<typename Arg>
    GT_FUNCTION
    constexpr auto operator()(Arg const& arg_) -> decltype((Arg()+Arg())/2.){
        return (arg_+arg_)/2.;
    }
};

namespace gdl{
    using namespace gt::expressions;
    struct residual{

        using rhs=gt::accessor<0, enumtype::in, gt::extent<> , 4>;
        using Ax=gt::accessor<1, enumtype::in, gt::extent<> , 4>;
        using res=gt::accessor<2, enumtype::inout, gt::extent<> , 4>;
        using arg_list=boost::mpl::vector<rhs, Ax, res> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index I;

            uint_t const n_dofs=eval.template get_storage_dim<3>(rhs());

            for(uint_t i=0; i<n_dofs; ++i)
                eval(res(I+i)) = eval( res(I+i) - Ax(I+i) + rhs(I+i));
        }
    };

    struct bc_functor{

        using bc=gt::accessor<0, enumtype::in, gt::extent<> , 4>;
        using result=gt::accessor<1, enumtype::inout, gt::extent<> , 4>;
        using arg_list=boost::mpl::vector<bc, result> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
        gt::dimension<4>::Index I;

        //assign the points on face 3 (x=0)
        //TODO hardcoded
        // eval(result()) = eval(bc());
        // eval(result(I+2)) = eval(bc(I+2));
        // eval(result(I+4)) = eval(bc(I+4));
        // eval(result(I+6)) = eval(bc(I+6));

        eval(result(I+1)) = eval(bc());
        eval(result(I+3)) = eval(bc(I+2));
        eval(result(I+5)) = eval(bc(I+4));
        eval(result(I+7)) = eval(bc(I+6));
        }
    };
}

int main( int argc, char ** argv){

    if(argc!=2){
        printf("usage: \n >> dg_advection <N>\n");
        exit(-666);
    }
    int it_ = atoi(argv[1]);

    //![definitions]
    using namespace gdl;
    using namespace gdl::enumtype;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4> >;
    using matrix_type=storage_t< matrix_storage_info_t >;

    using bd_matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4,5> >; //last dimension is tha face
    using bd_matrix_type=storage_t< bd_matrix_storage_info_t >;

    static const ushort_t order=1;
    using geo_map=reference_element<order, Lagrange, Hexa>;
    using cub=cubature<geo_map::order()+1, geo_map::shape()>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    geo_t fe_;
    fe_.compute(Intrepid::OPERATOR_GRAD);
    fe_.compute(Intrepid::OPERATOR_VALUE);

    //boundary
    using bd_cub_t = intrepid::boundary_cub<geo_map, cub::cubDegree>;
    using bd_discr_t = intrepid::boundary_discr<bd_cub_t>;
    bd_cub_t bd_cub_;

    bd_discr_t bd_discr_(bd_cub_, 0, 1, 2, 3, 4, 5);//face ordinals

    bd_discr_.compute(Intrepid::OPERATOR_GRAD, fe_.get_ordering());
    bd_discr_.compute(Intrepid::OPERATOR_VALUE, fe_.get_ordering());

    //![boundary]

    using as_base=assembly_base<geo_t>;
    using as=assembly<geo_t>;
    using as_bd=bd_assembly<bd_discr_t>;

    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=4;
    auto d2=4;
    auto d3=4;

    geo_t geo_;
    geo_.compute(Intrepid::OPERATOR_GRAD);
    geo_.compute(Intrepid::OPERATOR_VALUE);
    //![as_instantiation]
    //constructing the integration tools on the boundary

    as_base assembler_base(d1,d2,d3);
    as assembler(geo_,d1,d2,d3);
    as_bd bd_assembler(bd_discr_,d1,d2,d3);

    // using domain_tuple_t = aggregator_type_tuple<as_bd, as, as_base>;
    // domain_tuple_t domain_tuple_ (bd_assembler, assembler, assembler_base);
    //![as_instantiation]

    //![grid]
    //constructing a structured cartesian grid

    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<geo_map::basis_cardinality(); point++)
                {
                    assembler_base.grid()( i,  j,  k,  point,  0)= (i + (1+geo_.grid()(point, 0, 0))/2.)/d1;
                    assembler_base.grid()( i,  j,  k,  point,  1)= (j + (1+geo_.grid()(point, 1, 0))/2.)/d2;
                    assembler_base.grid()( i,  j,  k,  point,  2)= (k + (1+geo_.grid()(point, 2, 0))/2.)/d3;
                }
    // ![grid]

    typedef BACKEND::storage_info<0, gridtools::layout_map<0,1,2> > meta_local_t;

    static const uint_t edge_nodes=tensor_product_element<1,order>::n_points::value;

    meta_local_t meta_local_(edge_nodes, edge_nodes, edge_nodes);

    io_rectilinear<as_base::grid_type, meta_local_t> io_(assembler_base.grid(), meta_local_);

    //![instantiation_stiffness]
    //defining the advection matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,geo_map::basis_cardinality(),geo_map::basis_cardinality());
    matrix_type advection_(meta_, 0., "advection");
    matrix_type mass_(meta_, 0., "mass");

    bd_matrix_storage_info_t bd_meta_(d1,d2,d3,geo_map::basis_cardinality(),geo_map::basis_cardinality(), 6/*faces*/);
    bd_matrix_type bd_mass_(bd_meta_, 0., "bd mass");

    using scalar_storage_info_t=storage_info< __COUNTER__, layout_tt<3>>;//TODO change: iterate on faces
    using vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4>>;//TODO change: iterate on faces
    using bd_scalar_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4>>;//TODO change: iterate on faces
    using bd_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4,5>>;//TODO change: iterate on faces

    using scalar_type=storage_t< scalar_storage_info_t >;
    using vector_type=storage_t< vector_storage_info_t >;
    using bd_scalar_type=storage_t< bd_scalar_storage_info_t >;
    using bd_vector_type=storage_t< bd_vector_storage_info_t >;

    scalar_storage_info_t scalar_meta_(d1,d2,d3,geo_map::basis_cardinality());
    vector_storage_info_t vec_meta_(d1,d2,d3,geo_map::basis_cardinality(), 3);
    bd_scalar_storage_info_t bd_scalar_meta_(d1,d2,d3, bd_discr_t::geo_map::basis_cardinality(), bd_discr_t::s_num_boundaries );
    bd_vector_storage_info_t bd_vector_meta_(d1,d2,d3, bd_discr_t::geo_map::basis_cardinality(), 3, bd_discr_t::s_num_boundaries );

    scalar_type u_(scalar_meta_, 0., "u");//initial solution
    vector_type beta_(vec_meta_, 0., "beta");

    // initialization
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
                for (uint_t point=0; point<geo_map::basis_cardinality(); point++)
                {
                    for (uint_t dim=0; dim<3; dim++)
                        if(dim==0)
                            beta_(i,j,k,point,dim)=-1.;
                    // if(i+j+k>4)
                    if( i==2 && j>3 && j<6 )
                        u_(i,j,k,point)=1.;//point;
                }

    io_.set_attribute_scalar<0>(u_, "initial condition");

    scalar_type result_(scalar_meta_, 0., "result");//new solution
    scalar_type unified_result_(scalar_meta_, 0., "unified result");//new solution

    bd_scalar_type bd_beta_n_(bd_scalar_meta_, 0., "beta*n");
    bd_vector_type normals_(bd_vector_meta_, 0., "normals");
    bd_vector_type flux_(bd_vector_meta_, 0., "flux");

    bd_matrix_type bd_mass_uv_(bd_meta_, 0., "mass uv");

    scalar_type rhs_(scalar_meta_, 0., "rhs");//zero rhs
    using bc_storage_info_t=storage_info< __COUNTER__, gt::layout_map<-1,0,1,2> >;
    using bc_storage_t = storage_t< bc_storage_info_t >;
    bc_storage_info_t bc_meta_(1,d2,d3, geo_map::basis_cardinality());
    bc_storage_t bc_(bc_meta_, 0.);

    //![placeholders]
    // defining the placeholder for the mass
    // typedef arg<domain_tuple_t::size, bd_matrix_type> p_bd_mass;
    // defining the placeholder for the local gradient of the element boundary face
    // typedef arg<domain_tuple_t::size+1, bd_discr_t::grad_storage_t> p_bd_dphi;

    // appending the placeholders to the list of placeholders already in place
    // auto domain=domain_tuple_.template domain
    //     <p_u, p_result , p_mass, p_advection, p_beta
    //      , p_phi, p_dphi, p_beta_n, p_normals, p_unified_result>
    //     ( u_, result_, mass_, advection_, beta_
    //       ,  geo_.val(), geo_.grad(), bd_beta_n_, normals_, unified_result_);

    typedef gt::arg<0, typename as_base::grid_type >       p_grid_points;
    typedef gt::arg<1, typename as::jacobian_type >   p_jac;
    typedef gt::arg<2, typename as::geometry_t::weights_storage_t >   p_weights;
    typedef gt::arg<3, typename as::storage_type >    p_jac_det;
    typedef gt::arg<4, typename as::jacobian_type >   p_jac_inv;
    typedef gt::arg<5, typename as::geometry_t::basis_function_storage_t> p_phi;
    typedef gt::arg<6, typename as::geometry_t::grad_storage_t> p_dphi;

    typedef gt::arg<7, typename as_bd::jacobian_type >       p_bd_jac;
    typedef gt::arg<8, typename as_bd::face_normals_type >                   p_normals;
    typedef gt::arg<9, typename as_bd::storage_type >        p_bd_measure;
    typedef gt::arg<10, typename as_bd::boundary_t::weights_storage_t> p_bd_weights;
    typedef gt::arg<11, typename as_bd::boundary_t::tangent_storage_t> p_ref_normals;
    typedef gt::arg<12, bd_matrix_type> p_bd_mass_uu;
    typedef gt::arg<13, bd_matrix_type> p_bd_mass_uv;
    typedef gt::arg<14, typename as_bd::boundary_t::basis_function_storage_t> p_bd_phi;
    typedef gt::arg<15, typename as_bd::boundary_t::grad_storage_t> p_bd_dphi;
    typedef gt::arg<16, bd_vector_type> p_flux;

    typedef gt::arg<17, scalar_type> p_u;
    typedef gt::arg<18, scalar_type> p_result;
    typedef gt::arg<19, matrix_type> p_mass;
    typedef gt::arg<20, matrix_type> p_advection;
    typedef gt::arg<21, vector_type> p_beta;
    typedef gt::arg<22, typename geo_t::basis_function_storage_t> p_phi_discr;
    typedef gt::arg<23, typename geo_t::grad_storage_t> p_dphi_discr;
    typedef gt::arg<24, bd_scalar_type> p_beta_n;
    typedef gt::arg<25, bd_vector_type> p_int_normals;

    typedef gt::arg<26, scalar_type> p_unified_result;

    typedef typename boost::mpl::vector<p_grid_points, p_jac, p_weights, p_jac_det, p_jac_inv, p_phi, p_dphi, p_bd_jac, p_normals, p_bd_measure, p_bd_weights, p_ref_normals, p_bd_mass_uu, p_bd_mass_uv, p_bd_phi, p_bd_dphi, p_flux, p_u, p_result , p_mass, p_advection, p_beta, p_phi_discr, p_dphi_discr, p_beta_n, p_int_normals, p_unified_result> mpl_list;

    gt::aggregator_type<mpl_list> domain(boost::fusion::make_vector(  &assembler_base.grid()
                                                              , &assembler.jac()
                                                              , &assembler.fe_backend().cub_weights()
                                                              , &assembler.jac_det()
                                                              , &assembler.jac_inv()
                                                              , &assembler.fe_backend().val()
                                                              , &assembler.fe_backend().grad()
                                                              , &bd_assembler.bd_jac()
                                                              , &bd_assembler.normals()
                                                              , &bd_assembler.bd_measure()
                                                              , &bd_assembler.bd_backend().bd_cub_weights()
                                                              , &bd_assembler.bd_backend().ref_normals()
                                                              , &bd_mass_
                                                              , &bd_mass_uv_
                                                              , &bd_assembler.bd_backend().val()
                                                              , &bd_assembler.bd_backend().grad()
                                                              , &flux_
                                                              , &u_
                                                              , &result_
                                                              , &mass_
                                                              , &advection_
                                                              , &beta_
                                                              , &geo_.val()
                                                              , &geo_.grad()
                                                              , &bd_beta_n_
                                                              , &normals_
                                                              , &unified_result_
                                         ));

    //![placeholders]

    auto coords=gt::grid<axis>({1u, 0u, 1u, (uint_t)d1-1, (uint_t)d1},
        {1u, 0u, 1u, (uint_t)d2-1u, (uint_t)d2});
    coords.value_list[0] = 1;
    coords.value_list[1] = d3-1;

    //![computation]
    auto compute_assembly=gt::make_computation< BACKEND >(
        domain, coords,
        gt::make_multistage
        (
            execute<forward>()

            // boundary fluxes

            //computes the jacobian in the boundary points of each element
            , gt::make_stage<functors::update_bd_jac<as_bd::boundary_t, Hexa> >(p_grid_points(), p_bd_dphi(), p_bd_jac())
            //computes the measure of the boundaries with codimension 1 (ok, faces)
            , gt::make_stage<functors::measure<as_bd::boundary_t, 1> >(p_bd_jac(), p_bd_measure())
            //computes the mass on the element boundaries
            , gt::make_stage<functors::bd_mass<as_bd::boundary_t, as_bd::bd_cub> >(p_bd_measure(), p_bd_weights(), p_bd_phi(), p_bd_phi(), p_bd_mass_uu())
            , gt::make_stage<functors::bd_mass_uv<as_bd::boundary_t, as_bd::bd_cub> >(p_bd_measure(), p_bd_weights(), p_bd_phi(), p_bd_phi(), p_bd_mass_uv())

            // Internal element

            //compute the Jacobian matrix
            , gt::make_stage<functors::update_jac<as::geometry_t, Hexa> >(p_grid_points(), p_dphi(), p_jac())
            // compute the measure (det(J))
            , gt::make_stage<functors::det<geo_t> >(p_jac(), p_jac_det())
            // compute the mass matrix
            , gt::make_stage< functors::mass >(p_jac_det(), p_weights(), p_phi_discr(), p_phi_discr(), p_mass()) //mass
            // compute the jacobian inverse
            , gt::make_stage<functors::inv<geo_t> >(p_jac(), p_jac_det(), p_jac_inv())
            // compute the advection matrix
            , gt::make_stage<functors::advection< geo_t, cub > >(p_jac_det(), p_jac_inv(), p_weights(), p_beta(), p_dphi_discr(), p_phi_discr(), p_advection()) //advection

            // computing flux/discretize

            // initialize result=0
            //, gt::make_stage< functors::assign<4,zero<int> >( p_result() )
            // compute the face normals: \f$ n=J*(\hat n) \f$
            , gt::make_stage<functors::compute_face_normals<as_bd::boundary_t> >(p_bd_jac(), p_ref_normals(), p_normals())
            // interpolate the normals \f$ n=\sum_i <n,\phi_i>\phi_i(x) \f$
            , gt::make_stage<functors::bd_integrate<as_bd::boundary_t> >(p_bd_phi(), p_bd_measure(), p_bd_weights(), p_normals(), p_int_normals())
            // project beta on the normal direction on the boundary \f$ \beta_n = M<\beta,n> \f$
            // note that beta is defined in the current space, so we take the scalar product with
            // the normals on the current configuration, i.e. \f$F\hat n\f$
            , gt::make_stage<functors::project_on_boundary>(p_beta(), p_int_normals(), p_bd_mass_uu(), p_beta_n())
            //, gt::make_stage<functors::upwind>(p_u(), p_beta_n(), p_bd_mass_uu(), p_bd_mass_uv(),  p_result())

            // Optional: assemble the result vector by summing the values on the element boundaries
            // , gt::make_stage< functors::assemble<geo_t> >( p_result(), p_result() )
            // for visualization: the result is replicated
            // , gt::make_stage< functors::uniform<geo_t> >( p_result(), p_result() )
            // , gt::make_stage< time_advance >(p_u(), p_result())
            ));

    compute_assembly->ready();
    compute_assembly->steady();
    compute_assembly->run();

    // //removing unused storage
    // assembler_base.grid().release();
    // assembler.jac().release();
    // assembler.fe_backend().cub_weights().release();
    // assembler.jac_det().release();
    // assembler.jac_inv().release();
    // assembler.fe_backend().val().release();
    // assembler.fe_backend().grad().release();
    // bd_assembler.bd_jac().release();
    // bd_assembler.normals().release();
    // bd_assembler.bd_measure().release();
    // bd_assembler.bd_backend().bd_cub_weights().release();
    // bd_assembler.bd_backend().ref_normals().release();
    // bd_assembler.bd_backend().val().release();
    // bd_assembler.bd_backend().grad().release();
    // beta_.release();
    // geo_.val().release();
    // geo_.grad().release();
    // normals_.release();
    // unified_result_.release();

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

    // scalar_type rhs_(scalar_meta_, 0., "rhs");//zero rhs

    typedef typename boost::mpl::vector< it::p_bd_mass_uu, it::p_bd_mass_uv, it::p_u, it::p_result, it::p_mass, it::p_advection, it::p_beta_n, it::p_rhs > mpl_list_iteration;

    gt::aggregator_type<mpl_list_iteration> domain_iteration(boost::fusion::make_vector( &bd_mass_
                                                                                 , &bd_mass_uv_
                                                                                 , &u_
                                                                                 , &result_
                                                                                 , &mass_
                                                                                 , &advection_
                                                                                 , &bd_beta_n_
                                                                                 , &rhs_
                                                         ));

    auto iteration=gt::make_computation< BACKEND >(
            domain_iteration, coords
            , gt::make_multistage
            (
                execute<forward>()
                , gt::make_stage< functors::assign<4, zero<int> > >( it::p_result() )
                // add the advection term: result+=A*u
                , gt::make_stage< functors::matvec>( it::p_advection(), it::p_u(),it::p_result() )
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
                , gt::make_stage<residual>(it::p_rhs(), it::p_result(), it::p_u()) //updating u = u - (Ax-rhs)
                )
        );

    auto coords_bc=gt::grid<axis>({0u,0u,0u,0u,1u},
        {1u, 0u, 1u, (uint_t)d2-1u, (uint_t)d2});
    coords_bc.value_list[0] = 1;
    coords_bc.value_list[1] = d3-1;

    /** boundary condition computation */
    struct bc {
        typedef  gt::arg<0, bc_storage_t > p_bc;
        typedef  gt::arg<1, scalar_type> p_result;
    };

    typedef typename boost::mpl::vector< bc::p_bc, bc::p_result> mpl_list_bc;

    gt::aggregator_type<mpl_list_bc> domain_bc(boost::fusion::make_vector(  &bc_
                                                                   ,&u_
                                           ));

    //initialization of the boundary condition
    for(uint_t j=0; j<d2; ++j)
        for(uint_t k=0; k<d3; ++k)
            for(uint_t dof=0; dof<geo_map::basis_cardinality(); ++dof)
                bc_(666, j, k, dof) = 1.;

    auto apply_bc_x0=gt::make_computation< BACKEND >(
        domain_bc, coords_bc,
        gt::make_multistage
        (
            execute<forward>()
            , gt::make_stage< bc_functor >( bc::p_bc(), bc::p_result() )
            )
        );

    int n_it_ = it_;

    iteration->ready();
    iteration->steady();

    apply_bc_x0->ready();
    apply_bc_x0->steady();

    for(int i=0; i<n_it_; ++i){ // Richardson iterations
        apply_bc_x0->run();
        iteration->run();
    }
    apply_bc_x0->run();

    compute_assembly->finalize();
    iteration->finalize();
    apply_bc_x0->finalize();

    // for(int i=0; i<d1; ++i)
    //     for(int j=0; j<d1; ++j)
    //         for(int k=0; k<d1; ++k)
    //             //for(int dof=0; dof<8; ++dof)
    //             for(int d=0; d<3; ++d)
    //             for(int face=0; face<6; ++face)
    // std::cout<<"dof"<< dof <<"result==>"<<result_(i,j,k,dof)<<"\n";
    //                std::cout<<"face: "<<face<<", dim:"<<d <<"==>"<<normals_(i,j,k,0,d,face)<<"\n";
    //std::cout<<"face: "<<face<<", dim:"<<d <<"==>"<<bd_beta_n_(i,j,k,0,face)<<"\n";

    // std::cout<<"face: "<<face<<"==>"<<bd_assembler.normals()(i,j,k,0,d,face)<<"\n";

    io_.set_information("Time");
    io_.set_attribute_scalar<0>(result_, "Ax");
    io_.set_attribute_scalar<0>(u_, "solution");
    // io_.set_attribute_vector_on_face<0>(face_vec, "normals");
    io_.write("grid");

    spy(mass_, "mass.txt");
    spy_vec(result_, "sol.txt");
    spy_vec(u_, "init.txt");
    spy(advection_, "advection.txt");
//![computation]
    // intrepid::test(assembler, bd_discr_, bd_mass_);
}
