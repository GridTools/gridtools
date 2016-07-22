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

#include <common/layout_map_metafunctions.hpp>

//! [assembly]
#include "../numerics/bd_assembly.hpp"
//! [assembly]
#include "../numerics/tensor_product_element.hpp"
#include "../functors/matvec.hpp"
#include "../functors/interpolate.hpp"
#include "../tools/io.hpp"
#include "boundary.hpp"

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
    struct bc_functor{

        using bc=gt::accessor<0, enumtype::in, gt::extent<> , 4>;
        using out=gt::accessor<1, enumtype::inout, gt::extent<> , 4>;
        using arg_list=boost::mpl::vector<bc, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            gt::dimension<4>::Index I;

            uint_t const n_points=eval.template get_storage_dim<3>(bc());

            //assign the points on the boundary layer of elements
            for(int i=0; i<n_points; ++i){
                eval(out(I+i)) = eval(bc(I+i));
            }
        }
    };
}

int main( int argc, char ** argv){

    if(argc!=2){
        printf("usage: \n >> legendre <N>\n");
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

    static const ushort_t order_geom=1;
    static const ushort_t order_discr=2;
    using geo_map=reference_element<order_geom, Lagrange, Hexa>;
    using discr_map=reference_element<order_discr, Legendre, Hexa>;
    //integrate exactly polyunomials of degree (discr_map::order*geo_map::order)
    using cub=cubature<(discr_map::order() * geo_map::order())+2, geo_map::shape()>;//overintegrating: few basis func are 0 on all quad points otherwise...
    using discr_t = intrepid::discretization<discr_map, cub>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    discr_t fe_;
    geo_t geo_;
    fe_.compute(Intrepid::OPERATOR_VALUE);
    fe_.compute(Intrepid::OPERATOR_GRAD);
    geo_.compute(Intrepid::OPERATOR_GRAD);

    //boundary
    using bd_geo_cub_t = intrepid::boundary_cub<geo_map, cub::cubDegree>;
    using bd_discr_cub_t = intrepid::boundary_cub<discr_map, cub::cubDegree>;
    using bd_geo_t = intrepid::boundary_discr<bd_geo_cub_t>;
    using bd_discr_t = intrepid::boundary_discr<bd_discr_cub_t>;
    bd_geo_cub_t bd_cub_geo_;
    bd_geo_t bd_geo_(bd_cub_geo_, 0, 1, 2, 3, 4, 5);//face ordinals

    bd_discr_cub_t bd_cub_discr_;
    bd_discr_t bd_discr_(bd_cub_discr_, 0, 1, 2, 3, 4, 5);//face ordinals

    bd_geo_.compute(Intrepid::OPERATOR_VALUE// , geo_.get_ordering()
        );
    bd_geo_.compute(Intrepid::OPERATOR_GRAD// , geo_.get_ordering()
        );

    bd_discr_.compute(Intrepid::OPERATOR_VALUE);

    //![boundary]

    using as_base=assembly_base<geo_t>;
    using as=assembly<geo_t>;
    using as_bd=bd_assembly<bd_geo_t>;

    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=8;
    auto d2=8;
    auto d3=8;

    //![as_instantiation]
    //constructing the integration tools on the boundary

    as_base assembler_base(d1,d2,d3);
    as assembler(geo_,d1,d2,d3);
    as_bd bd_assembler(bd_geo_,d1,d2,d3);

    // using domain_tuple_t = gt::aggregator_type_tuple<as_bd, as, as_base>;
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

    static const uint_t edge_nodes=tensor_product_element<1,order_geom>::n_points::value;

    meta_local_t meta_local_(edge_nodes, edge_nodes, edge_nodes);

    // io_rectilinear_qpoints<as_base::grid_type, meta_local_t> io_(assembler_base.grid(), meta_local_);
    io_rectilinear_qpoints<as_base::grid_type, discr_t::cub_points_storage_t, gt::static_ushort<cub::cubDegree> > io_(assembler_base.grid(), fe_.get_cub_points());

    //![instantiation_stiffness]
    //defining the advection matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_(d1,d2,d3,discr_map::basis_cardinality(),discr_map::basis_cardinality());
    matrix_type advection_(meta_, 0., "advection");
    matrix_type mass_(meta_, 0., "mass");

    bd_matrix_storage_info_t bd_meta_(d1,d2,d3,discr_map::basis_cardinality(),discr_map::basis_cardinality(), 6/*faces*/);
    bd_matrix_type bd_mass_(bd_meta_, 0., "bd mass");

    using scalar_storage_info_t=storage_info< __COUNTER__, layout_tt<3>>;//TODO change: iterate on faces
    using vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4>>;//TODO change: iterate on faces
    using bd_scalar_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4>>;//TODO change: iterate on faces
    using bd_vector_storage_info_t=storage_info< __COUNTER__, layout_tt<3,4,5>>;//TODO change: iterate on faces

    using scalar_type=storage_t< scalar_storage_info_t >;
    using vector_type=storage_t< vector_storage_info_t >;
    using bd_scalar_type=storage_t< bd_scalar_storage_info_t >;
    using bd_vector_type=storage_t< bd_vector_storage_info_t >;

    scalar_storage_info_t scalar_meta_(d1,d2,d3,discr_map::basis_cardinality());
    vector_storage_info_t vec_meta_(d1,d2,d3,discr_map::basis_cardinality(), 3);
    bd_scalar_storage_info_t bd_scalar_meta_(d1,d2,d3, discr_map::basis_cardinality(), bd_discr_t::s_num_boundaries );
    bd_vector_storage_info_t bd_vector_meta_(d1,d2,d3, discr_map::basis_cardinality(), 3, bd_discr_t::s_num_boundaries );

    scalar_type u_(scalar_meta_, 0., "u");//initial solution

    using physical_scalar_storage_info_t = storage_info< __COUNTER__, layout_tt<3> >;
    using physical_scalar_storage_type = storage_t<physical_scalar_storage_info_t>;
    using physical_vec_storage_info_t =storage_info< __COUNTER__, layout_tt<3,4> >;
    using physical_vec_storage_type = storage_t<physical_vec_storage_info_t>;

    physical_vec_storage_info_t physical_vec_info_(d1,d2,d3,cub::numCubPoints(),3);

    vector_type beta_interp_(vec_meta_, 0., "beta");
    physical_vec_storage_type beta_phys_(physical_vec_info_, "beta interp");

    // initialization
    for (uint_t i=0; i<d1; i++)
        for (uint_t j=0; j<d2; j++)
            for (uint_t k=0; k<d3; k++)
            {
                for (uint_t point=0; point<cub::numCubPoints(); point++)
                {
                    auto norm=std::sqrt(
                        gt::gt_pow<2>::apply(assembler_base.grid()( i,  j,  k,  0,  0) + assembler.fe_backend().cub_points()(point, 0, 0))
                        +
                        gt::gt_pow<2>::apply(assembler_base.grid()( i,  j,  k,  0,  1) + assembler.fe_backend().cub_points()(point, 1, 0)));

                    if(norm)
                    {
                        beta_phys_(i,j,k,point,0)=-(assembler_base.grid()( i,  j,  k,  0,  1) + assembler.fe_backend().cub_points()(point, 0, 0))/norm;
                        beta_phys_(i,j,k,point,1)=(assembler_base.grid()( i,  j,  k,  0,  0) + assembler.fe_backend().cub_points()(point, 1, 0))/norm;
                    }
                    else{
                        beta_phys_(i,j,k,point,0)=0.;
                        beta_phys_(i,j,k,point,1)=0.;
                    }
                    beta_phys_(i,j,k,point,2)=0.;
                }
                for (uint_t point=0; point<discr_map::basis_cardinality(); point++)
                {
                    // if(i+j+k>4)
                    if( i==2 && j>3 && j<6 )
                        u_(i,j,k,point)=1.;//point;
                }
            }

    // io_.set_attribute_scalar<0>(u_, "initial condition");

    scalar_type result_(scalar_meta_, 0., "result");//new solution
    // scalar_type unified_result_(scalar_meta_, 0., "unified result");//new solution

    bd_scalar_type bd_beta_n_(bd_scalar_meta_, 0., "beta*n");
    bd_vector_type normals_(bd_vector_meta_, 0., "normals");
    bd_vector_type flux_(bd_vector_meta_, 0., "flux");

    bd_matrix_type bd_mass_uv_(bd_meta_, 0., "mass uv");

    scalar_type rhs_(scalar_meta_, 0., "rhs");//zero rhs

    //![placeholders]
    // defining the placeholder for the mass
    // typedef gt::arg<domain_tuple_t::size, bd_matrix_type> p_bd_mass;
    // defining the placeholder for the local gradient of the element boundary face
    // typedef gt::arg<domain_tuple_t::size+1, bd_discr_t::grad_storage_t> p_bd_dphi;

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
    typedef gt::arg<15, bd_vector_type> p_flux;

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

    gt::aggregator_type<mpl_list> domain(boost::fusion::make_vector(  &assembler_base.grid()
                                                              , &assembler.jac()
                                                              , &assembler.fe_backend().cub_weights()
                                                              , &assembler.jac_det()
                                                              , &assembler.jac_inv()
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
                                                              , &beta_phys_
                                                              , &beta_interp_
                                                              , &fe_.val()
                                                              , &fe_.grad()
                                                              , &bd_beta_n_
                                                              , &normals_
                                     ));

    //![placeholders]

    auto coords_all=gt::grid<axis>({0u, 0u, 0u, (uint_t)d1-1, (uint_t)d1},
        {0u, 0u, 0u, (uint_t)d2-1u, (uint_t)d2});
    coords_all.value_list[0] = 0;
    coords_all.value_list[1] = d3-1;

    auto compute_jacobian=gt::make_computation< BACKEND >(
        domain, coords_all
        , gt::make_multistage(
            execute<forward>()
            //compute the Jacobian matrix
            , gt::make_stage<functors::update_jac<as::geometry_t, Hexa> >(p_grid_points(), p_dphi(), p_jac())
            // compute the measure (det(J))
            , gt::make_stage<functors::det<geo_t> >(p_jac(), p_jac_det())
            // compute the jacobian inverse
            , gt::make_stage<functors::inv<geo_t> >(p_jac(), p_jac_det(), p_jac_inv())
            ));

    auto coords=gt::grid<axis>({1u, 0u, 1u, (uint_t)d1-1, (uint_t)d1},
        {1u, 0u, 1u, (uint_t)d2-1u, (uint_t)d2});
    coords.value_list[0] = 1;
    coords.value_list[1] = d3-1;

    //![computation]
    auto compute_assembly=gt::make_computation< BACKEND >(
        domain, coords
        , gt::make_multistage(
            execute<forward>()

            // boundary fluxes

            //computes the jacobian in the boundary points of each element *
            , gt::make_stage<functors::update_bd_jac<as_bd::boundary_t, Hexa> >(p_grid_points(), p_bd_dphi(), p_bd_jac())
            // //computes the measure of the boundaries with codimension 1 (ok, faces)
            // , gt::make_stage<functors::measure<as_bd::boundary_t, 1> >(p_bd_jac(), p_bd_measure())
            // computes the mass on the element boundaries*
            , gt::make_stage<functors::bd_mass<as_bd::boundary_t, as_bd::bd_cub> >(p_bd_measure(), p_bd_weights(), p_bd_phi(), p_bd_phi(), p_bd_mass_uu())
            //*
            , gt::make_stage<functors::bd_mass_uv<as_bd::boundary_t, as_bd::bd_cub> >(p_bd_measure(), p_bd_weights(), p_bd_phi(), p_bd_phi(), p_bd_mass_uv())

            //Internal element

            // interpolate beta *
            , gt::make_stage< functors::transform_vec >( p_jac_det(), p_weights(), p_phi_discr(), p_beta_phys(), p_beta_interp() )

            // compute the mass matrix
            // , gt::make_stage< functors::mass >(p_jac_det(), p_weights(), p_phi_discr(), p_phi_discr(), p_mass()) //mass
            // compute the advection matrix
            , gt::make_stage<functors::advection< geo_t, cub > >(p_jac_det(), p_jac_inv(), p_weights(), p_beta_interp(), p_dphi_discr(), p_phi_discr(), p_advection()) //advection

            // computing flux/discretize

            // initialize result=0
            //, gt::make_stage< functors::assign<4,zero<int> > >( p_result() )
            // compute the face normals: \f$ n=J*(\hat n) \f$
            // , gt::make_stage<functors::compute_face_normals<as_bd::boundary_t> >(p_bd_jac(), p_ref_normals(), p_normals())
            // // interpolate the normals \f$ n=\sum_i <n,\phi_i>\phi_i(x) \f$
            // , gt::make_stage<functors::bd_integrate<as_bd::boundary_t> >(p_bd_phi(), p_bd_measure(), p_bd_weights(), p_normals(), p_int_normals())
            // // project beta on the normal direction on the boundary \f$ \beta_n = M<\beta,n> \f$
            // // note that beta is defined in the current space, so we take the scalar product with
            // // the normals on the current configuration, i.e. \f$F\hat n\f$
             // , gt::make_stage<functors::project_on_boundary>(p_beta_interp(), p_int_normals(), p_bd_mass_uu(), p_beta_n())
            // //, gt::make_stage<functors::upwind>(p_u(), p_beta_n(), p_bd_mass_uu(), p_bd_mass_uv(),  p_result())

            // Optional: assemble the result vector by summing the values on the element boundaries
            // , gt::make_stage< functors::assemble<geo_t> >( p_result(), p_result() )
            // for visualization: the result is replicated
            // , gt::make_stage< functors::uniform<geo_t> >( p_result(), p_result() )
            // , gt::make_stage< time_advance >(p_u(), p_result())
            ));

    compute_jacobian->ready();
    compute_jacobian->steady();
    compute_jacobian->run();
    compute_jacobian->finalize();

    compute_assembly->ready();
    compute_assembly->steady();
    compute_assembly->run();

    compute_assembly->finalize();

    //removing unused storage
    assembler_base.grid().release();
    assembler.jac().release();
    // assembler.fe_backend().cub_weights().release();
    // assembler.jac_det().release();
    assembler.jac_inv().release();
    // assembler.fe_backend().val().release();
    assembler.fe_backend().grad().release();
    bd_assembler.bd_jac().release();
    bd_assembler.normals().release();
    bd_assembler.bd_measure().release();
    bd_assembler.bd_backend().bd_cub_weights().release();
    bd_assembler.bd_backend().ref_normals().release();
    bd_assembler.bd_backend().val().release();
    bd_assembler.bd_backend().grad().release();
    beta_phys_.release();
    // geo_.val().release();
    geo_.grad().release();
    normals_.release();

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
         , gt::make_multistage (
             execute<forward>()
             , gt::make_stage< functors::assign<4,zero<int> > >( it::p_result() )
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

    // boundary condition computation

    auto coords_low=gt::grid<axis>({1u,0u,1u,d1-1u,d1},
            {1u, 0u, 1u, (uint_t)d2-1u, (uint_t)d2});
    coords_low.value_list[0] = 0;
    coords_low.value_list[1] = 0;


    using bc_storage_info_t=storage_info< __COUNTER__, gt::layout_map< -1,0,1,2 > >;
    using bc_storage_t = storage_t< bc_storage_info_t >;

    using bc_tr_storage_info_t=storage_info< __COUNTER__, gt::layout_map< -1,0,1,2 > >;
    using bc_tr_storage_t = storage_t< bc_tr_storage_info_t >;

    bc_storage_info_t bc_low_meta_(1,d2,d3, bd_geo_cub_t::bd_cub::numCubPoints());
    bc_storage_t bc_low_(bc_low_meta_, 0.);

    bc_tr_storage_info_t tr_bc_low_meta_(1,d2,d3, discr_map::basis_cardinality());
    bc_tr_storage_t tr_bc_low_(tr_bc_low_meta_, 0.);

    // bc_apply< as, discr_t, bc_functor, functors::upwind> bc_apply_(assembler, fe_);
    // auto bc_compute_low = bc_apply_.compute(coords_low, bc_low_, tr_bc_low_);
    // auto bc_apply_low = bc_apply_.template apply(coords_low
    //                                              ,tr_bc_low_
    //                                              ,result_
    //                                              ,bd_beta_n_
    //                                              ,bd_mass_
    //                                              ,bd_mass_uv_
    //                                              ,u_
    //     );

        struct transform_bc {
            typedef  gt::arg<0, typename as::storage_type >    p_jac_det;
            typedef  gt::arg<1, typename as::geometry_t::weights_storage_t >   p_weights;
            typedef  gt::arg<2, typename discr_t::basis_function_storage_t> p_phi;
            typedef  gt::arg<3, bc_storage_t > p_bc;
            typedef  gt::arg<4, bc_tr_storage_t > p_bc_integrated;
        };

        //adding an extra index at the end of the layout_map
        // using bc_storage_info_t=storage_info< __COUNTER__, gt::layout_map_union<Layout, gt::layout_map<0> > >;
        // using bc_storage_t = storage_t< bc_storage_info_t >;

        typedef typename boost::mpl::vector< typename transform_bc::p_jac_det, typename transform_bc::p_weights, typename transform_bc::p_phi, typename transform_bc::p_bc, typename transform_bc::p_bc_integrated> mpl_list_transform_bc;

        gt::aggregator_type<mpl_list_transform_bc> domain_transform_bc(
            boost::fusion::make_vector(
                &assembler.jac_det()
                ,&assembler.fe_backend().cub_weights()
                ,&fe_.val()
                ,&bc_low_
                ,&tr_bc_low_
                ));

        auto bc_compute_low=gt::make_computation< BACKEND >(
            domain_transform_bc, coords_low
            , gt::make_multistage(
                execute<forward>()
                , gt::make_stage< bc_functor >( typename transform_bc::p_bc(), typename transform_bc::p_bc() )
                , gt::make_stage< functors::transform >( typename transform_bc::p_jac_det(), typename transform_bc::p_weights(), typename transform_bc::p_phi(), typename transform_bc::p_bc(), typename transform_bc::p_bc_integrated() )
                )
            );

        struct bc{
            typedef  gt::arg<0, bc_tr_storage_t > p_bc;
            typedef  gt::arg<1, scalar_type > p_result;
            typedef  gt::arg<2, bd_scalar_type > p_beta_n;
            typedef  gt::arg<3, bd_matrix_type > p_bd_mass_uu;
            typedef  gt::arg<4, bd_matrix_type > p_bd_mass_uv;
        };

        typedef typename boost::mpl::vector< typename bc::p_bc, typename bc::p_result, typename bc::p_beta_n, typename bc::p_bd_mass_uu, typename bc::p_bd_mass_uv> mpl_list_bc;

        gt::aggregator_type<mpl_list_bc> domain_apply_bc(boost::fusion::make_vector(
                                                         &tr_bc_low_
                                                         ,&result_
                                                         ,&bd_beta_n_
                                                         ,&bd_mass_
                                                         ,&bd_mass_uv_
                                                         ));

        auto bc_apply_low=gt::make_computation< BACKEND >(
            domain_apply_bc, coords_low
            , gt::make_multistage(
                execute<forward>()
                , gt::make_stage< functors::upwind >(typename bc::p_bc(), typename bc::p_beta_n(), typename bc::p_bd_mass_uu(), typename bc::p_bd_mass_uv(),  typename bc::p_result())
                )
            );


    // auto coords_right=gt::grid<axis>({1u,0u,1u,d1-1u,d1},
    //         {d2-1u, 0u, d2-1u, (uint_t)d2-1u, (uint_t)d2});
    // coords_right.value_list[0] = 0;
    // coords_right.value_list[1] = d3-1;

    // using bc_right_storage_info_t=storage_info< __COUNTER__, gt::layout_map< 0,-1,1,2 > >;
    // using bc_right_storage_t = storage_t< bc_right_storage_info_t >;

    // using bc_right_tr_storage_info_t=storage_info< __COUNTER__, gt::layout_map< 0,-1,1,2 > >;
    // using bc_right_tr_storage_t = storage_t< bc_right_tr_storage_info_t >;

    // bc_storage_info_t bc_right_meta_(d1,1,d3, bd_geo_cub_t::bd_cub::numCubPoints());
    // bc_storage_t bc_right_(bc_right_meta_, 0.);

    // bc_tr_storage_info_t tr_bc_right_meta_(d1,1,d3, discr_map::basis_cardinality());
    // bc_tr_storage_t tr_bc_right_(tr_bc_right_meta_, 0.);

    // auto bc_compute_right = bc_apply_.compute(coords_right, bc_right_, tr_bc_right_);
    // auto bc_apply_right = bc_apply_.template apply(coords_right
    //                                                  ,tr_bc_right_
    //                                                  ,result_
    //                                                  ,bd_beta_n_
    //                                                  ,bd_mass_
    //                                                  ,bd_mass_uv_
    //                                                  ,u_
    //     );


    //initialization of the boundary condition
    for(uint_t j=0; j<d2; ++j)
        for(uint_t k=0; k<d3; ++k)
            for(uint_t dof=0; dof<bd_geo_cub_t::bd_cub::numCubPoints(); ++dof)
            {
                if(j<d2+1)
                    bc_low_(666, j, k, dof) = 1.;
                else
                    bc_low_(666, j, k, dof) = 0.;
                // bc_right_( j, 666, k, dof) = 1.;
            }

    /* end of boundary conditions */

    int n_it_ = it_;

    iteration->ready();
    iteration->steady();

    // bc_compute_low->ready();
    // bc_apply_low->ready();
    // bc_compute_low->steady();
    // bc_apply_low->steady();

    for(int i=0; i<n_it_; ++i){ // Richardson iterations
        // bc_apply_low->run();
        iteration->run();
    }
    // bc_compute_low->run();
    // bc_apply_low->run();

    // for(int i=0; i<d1; ++i)
    //     for(int j=0; j<d1; ++j)
    //         for(int k=0; k<d1; ++k)
    //             //for(int dof=0; dof<8; ++dof)
    //             for(int d=0; d<3; ++d)
    //             for(int face=0; face<6; ++face)
    // std::cout<<"dof"<< dof <<"result==>"<<result_(i,j,k,dof)<<"\n";
    //                std::cout<<"face: "<<face<<", dim:"<<d <<"==>"<<normals_(i,j,k,0,d,face)<<"\n";
    // std::cout<<"face: "<<face<<", dim:"<<d <<"==>"<<bd_beta_n_(i,j,k,0,face)<<"\n";

    // std::cout<<"face: "<<face<<"==>"<<bd_assembler.normals()(i,j,k,0,d,face)<<"\n";

    io_.set_information("Time");
    // io_.set_attribute_scalar<0>(result_, "Ax");
    // io_.set_attribute_scalar<0>(u_, "solution");
    // io_.set_attribute_vector_on_face<0>(face_vec, "normals");
    io_.write("grid");

    // for(int i=0; i<d1; ++i)
    //     for(int j=0; j<d2; ++j)
    //         for(int k=0; k<d3; ++k)
    //             for(int l=0; l<10; ++l)
    //                 std::cout<< mass_(i,j,k,l,l)<<"\n";

    // spy(mass_, "mass.txt");

//     spy_vec(result_, "sol.txt");
//     spy_vec(u_, "init.txt");

    // spy(advection_, "advection.txt");

    struct evaluate{

        typedef  gt::arg<0, typename discr_t::basis_function_storage_t> p_phi;
        typedef  gt::arg<1, scalar_type> p_result;
        typedef  gt::arg<2, physical_scalar_storage_type
                     > p_result_interpolated;
        typedef  gt::arg<3, typename as::storage_type >    p_jac_det;
        typedef  gt::arg<4, typename as::geometry_t::weights_storage_t >   p_weights;
    };

    physical_scalar_storage_info_t physical_scalar_storage_info_(d1,d2,d3,cub::numCubPoints());
    physical_scalar_storage_type result_interpolated_(physical_scalar_storage_info_, 0., "interpolated result");

    typedef typename boost::mpl::vector< evaluate::p_phi, evaluate::p_result, evaluate::p_result_interpolated, evaluate::p_jac_det, evaluate::p_weights> mpl_list_interp;

    gt::aggregator_type<mpl_list_interp> domain_interp(boost::fusion::make_vector(
                                                   &fe_.val()
                                                   ,&result_
                                                   ,&result_interpolated_
                                                   ,&assembler.jac_det()
                                                   ,&assembler.fe_backend().cub_weights()
                        ));

    auto evaluation_=gt::make_computation< BACKEND >(
        domain_interp, coords
        , gt::make_multistage(
            execute<forward>()
            , gt::make_stage< functors::evaluate >( evaluate::p_phi(), evaluate::p_result(), evaluate::p_weights(), evaluate::p_jac_det(), evaluate::p_result_interpolated() )
            )
        );

    evaluation_->ready();
    evaluation_->steady();
    evaluation_->run();
    evaluation_->finalize();
    compute_assembly->finalize();
    iteration->finalize();
    // bc_compute_low->finalize();
    // bc_apply_low->finalize();

    io_.set_attribute_scalar<0>(result_interpolated_, "solution");

// //![computation]
//    // intrepid::test(assembler, bd_discr_, bd_mass_);
}
