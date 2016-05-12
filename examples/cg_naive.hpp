#pragma once

//disable pedantic mode for the global accessor
#define PEDANTIC_DISABLED

#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/backend.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>
#include <stencil-composition/reductions/reductions.hpp>

#include <storage/partitioner_trivial.hpp>
#include <storage/parallel_storage.hpp>

#include <communication/low-level/proc_grids_3D.hpp>
#include <communication/halo_exchange.hpp>
#include <boundary-conditions/apply.hpp>

#include <boost/timer/timer.hpp>
#include "cg.h"

//time t is in ns, returns MFLOPS
inline double MFLOPS(int numops, int X, int Y, int Z, int NT, int t) { return (double)numops*X*Y*Z*NT*1000/t; }
inline double MLUPS(int X, int Y, int Z, int NT, int t) { return (double)X*Y*Z*NT*1000/t; }

// domain function
inline double f(double xi, double yi, double zi){
    return 2.*(cos(xi+yi) - (1.+xi)*sin(xi+yi));
}

// boundary function
inline double g(double xi, double yi, double zi) {
    return (xi+1.)*sin(xi+yi);
}

// solution
inline double u(double xi, double yi, double zi) {
    return g(xi, yi, zi);
}

/*
  @file This file shows an implementation of Conjugate Gradient solver.

  7-point constant-coefficient isotropic stencil in three dimensions, with symmetry
  is used to implement matrix-free matrix-vector product. The matrix has a constant
  structure arising from finite element discretization.

  Regular domain x in 0..1 is discretized, the step size h = 1/(n+1)
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

namespace cg_naive{

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

// This is the definition of the special regions in the "vertical" direction
// The K dimension can be split into logical splitters. They lie in between array elements.
// A level represent an element on the third coordinate, such that level<2,1> represent
// the array index following splitter<2>, while level<2,-1> represents the array value
// before splitter<2>. Absolute placement of splitters is determined below.
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

/** @brief
    Parallel copy of a domain done on the backend.
    @param out Destination domain.
    @param int Source domain.
*/
struct copy_functor {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in> in;
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        eval(out{}) = eval(in{});
    }
};

/** @brief
    1st order in time, 7-point constant-coefficient isotropic stencil in 3D, with symmetry.

    @param in Source vector.
    @return Matrix-vector product out = A*in
*/
struct d3point7{
    typedef accessor<0, enumtype::inout, extent<0,0,0,0> > out;
    typedef accessor<1, enumtype::in, extent<-1,1,-1,1> > in; // this says to access 6 neighbors
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out{}) = 6.0 * dom(in{})
                    - (dom(in{x(-1)})+dom(in{x(+1)}))
                    - (dom(in{y(-1)})+dom(in{y(+1)}))
                    - (dom(in{z(-1)})+dom(in{z(+1)}));
    }
};

/** @brief
    Performs element-wise multiplication of the elements from the input grids

    @param a Source vector.
    @param b Source vector.
    @return Element-wise product out = a*b
*/
struct product_functor{
    typedef accessor<0, enumtype::inout, extent<0,0,0,0> > out;
    typedef accessor<1, enumtype::in, extent<0,0,0,0> > a;
    typedef accessor<2, enumtype::in, extent<0,0,0,0> > b;
    typedef boost::mpl::vector<out, a, b> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out{}) = dom(a{}) * dom(b{});
    }
};

/** @brief
    Provides access to elements of the grid

    @param in Source vector
*/
struct reduction_functor {

    typedef accessor< 0, enumtype::in > in;
    typedef boost::mpl::vector< in > arg_list;

    template < typename Evaluation >
    GT_FUNCTION
    static float_type Do(Evaluation const &eval, x_interval) {
        return eval(in());
    }
};

/** @brief generic argument type
   Minimal interface for parameter to be passed as an argument to the user functor.
*/
struct parameter : clonable_to_gpu<parameter> {

    parameter(double t=0.)
        :
        my_value(t)
    {}

    //device copy constructor
    __device__ parameter(const parameter& other) :
        my_value(other.my_value)
    {}

    typedef parameter super;
    typedef parameter* iterator_type;
    typedef parameter value_type; //TODO remove
    static const ushort_t field_dimensions=1; //TODO remove

    double my_value;

    void setValue(double v) {my_value = v;}
    double getValue() const {return my_value;}

    template<typename ID>
    parameter * access_value() const {return const_cast<parameter*>(this);}
};

/** @brief
    Stencil implementing addition of two grids

    @param a Source domain.
    @param b Source domain.
    @param c Destination domain.
    @param alpha Scalar.
    @return c = a + alpha * b
*/
struct add_functor{
    typedef accessor<0, enumtype::inout, extent<0,0,0,0> > c;
    typedef accessor<1, enumtype::in, extent<0,0,0,0> > a;
    typedef accessor<2, enumtype::in, extent<0,0,0,0> > b;
    typedef global_accessor<3, enumtype::in> alpha;
    typedef boost::mpl::vector<c,a,b> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(c{}) = dom(a{}) + dom(alpha{})->getValue() * dom(b{});
    }
};

/** @brief
    Implementation of the Dirichlet boundary conditions.
*/
template <typename Partitioner>
struct boundary_conditions {
    Partitioner const& m_partitioner; // info about domain partitioning
    double h; // step size

    boundary_conditions(Partitioner const& p, double step)
        : m_partitioner(p), h(step)
    {}

    // DataField_x are fields that are passed in the application of boundary condition
    template <typename Direction, typename DataField0, typename DataField1, typename DataField2, typename DataField3>
    GT_FUNCTION
    void operator()(Direction,
                    DataField0 & data_field0,
                    DataField1 & data_field1,
                    DataField2 & data_field2,
                    DataField3 & data_field3,
                    uint_t i, uint_t j, uint_t k) const {
        // get global indices on the boundary
        size_t I = m_partitioner.get_low_bound(0) + i;
        size_t J = m_partitioner.get_low_bound(1) + j;
        size_t K = m_partitioner.get_low_bound(2) + k;
        data_field0(i,j,k) = 0;//g(h*I, h*J, h*K); //x //TODO 
        data_field1(i,j,k) = 0;//g(h*I, h*J, h*K); //d
        data_field2(i,j,k) = 0;//g(h*I, h*J, h*K); //xNew //TODO
        data_field3(i,j,k) = 0;//g(h*I, h*J, h*K); //dNew
    }
};
/*******************************************************************************/
/*******************************************************************************/

bool solver(uint_t xdim, uint_t ydim, uint_t zdim, uint_t nt) {

    // initialize MPI
    gridtools::GCL_Init();

    // domain is encapsulated in boundary layer from both sides in each dimension,
    // these are just inner domain dimensions
    uint_t d1 = xdim;
    uint_t d2 = ydim;
    uint_t d3 = zdim;
    uint_t TIME_STEPS = nt;

    // enforce square domain
    if (!(xdim==ydim && ydim==zdim)) {
        if(PID==0) printf("Please run with dimensions X=Y=Z\n");
        return false;
    }

    // step size, add +2 for boundary layer
    double h = 1./(xdim+2+1);
    double h2 = h*h;

    if(PID == 0){
        printf("Running for %d x %d x %d, %d iterations\n", xdim+2, ydim+2, zdim+2, nt);
        // printf("Step size: %f\n", h);
    }

#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, GRIDBACKEND, Block >
#else
#define BACKEND backend<Host, GRIDBACKEND, Naive >
#endif

    //--------------------------------------------------------------------------
    // Create processor grid
    array<int, 3> dimensions{0,0,0};
    MPI_3D_process_grid_t<3>::dims_create(PROCS, 2, dimensions);
    dimensions[2]=1;

    // Prepare types for the data storage
    //                   strides  1 x xy
    //                      dims  x y z
    typedef gridtools::layout_map<0,1,2> layout_t;
    typedef gridtools::BACKEND::storage_info<0, layout_t> metadata_t;
    typedef gridtools::BACKEND::storage_type<float_type, metadata_t >::type storage_type;
    typedef storage_type::pointer_type pointer_type;

    typedef gridtools::halo_exchange_dynamic_ut<layout_t,
                                                gridtools::layout_map<0, 1, 2>,
                                                pointer_type::pointee_t,
                                                MPI_3D_process_grid_t<3> ,
                                                gridtools::gcl_cpu,
                                                gridtools::version_manual> pattern_type;

    pattern_type he(pattern_type::grid_type::period_type(false, false, false), GCL_WORLD, &dimensions);

    // 3D distributed storage
    array<ushort_t, 3> padding{1,1,0}; // global halo, 1-Dirichlet, 0-Neumann
    array<ushort_t, 3> halo{1,1,1}; // number of layers to communicate to neighboring processes
    typedef partitioner_trivial<cell_topology<topology::cartesian<layout_map<0,1,2> > >, pattern_type::grid_type> partitioner_t;
    partitioner_t part(he.comm(), halo, padding);
    parallel_storage_info<metadata_t, partitioner_t> meta_(part, d1, d2, d3);
    auto metadata_=meta_.get_metadata();

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after.
    // Iteration space is defined within axis.
    // K dimension not partitioned.
    gridtools::grid<axis, partitioner_t> coords3d7pt(part, meta_);
    coords3d7pt.value_list[0] = 1; //specifying index of the splitter<0,-1>
    coords3d7pt.value_list[1] = d3; //specifying index of the splitter<1,-1>

    //--------------------------------------------------------------------------
    // Definition of the actual data fields that are used for input/output

    storage_type b    (metadata_, 1., "RHS vector");
    storage_type x    (metadata_, 0., "Solution vector t");
    storage_type xNew (metadata_, 0., "Solution vector t+1");
    storage_type Ax   (metadata_, 0., "Vector Ax at time t");
    storage_type r    (metadata_, 0., "Residual at time t");
    storage_type rNew (metadata_, 0., "Residual at time t+1");
    storage_type d    (metadata_, 0., "Direction vector at time t");
    storage_type dNew (metadata_, 0., "Direction vector at time t+1");
    storage_type Ad   (metadata_, 0., "Projected direction vector");
    storage_type tmp  (metadata_, 0., "Temporary storage");

    // Pointers to data-fields are swapped at each time-iteration
    storage_type *ptr_x = &x, *ptr_xNew = &xNew;
    storage_type *ptr_r = &r, *ptr_rNew = &rNew;
    storage_type *ptr_d = &d, *ptr_dNew = &dNew;

    // Scalar parameters
    parameter alpha; //step length
    parameter beta; //orthogonalization parameter

    // set up halo
    he.add_halo<0>(meta_.template get_halo_gcl<0>());
    he.add_halo<1>(meta_.template get_halo_gcl<1>());
    he.add_halo<2>(meta_.template get_halo_gcl<2>());
    he.setup(2);

    // get global offsets
    size_t I = meta_.get_low_bound(0);
    size_t J = meta_.get_low_bound(1);
    size_t K = meta_.get_low_bound(2);

    // initialize the RHS vector domain
    for(uint_t i=0; i<metadata_.template dims<0>(); ++i)
        for(uint_t j=0; j<metadata_.template dims<1>(); ++j)
            for(uint_t k=0; k<metadata_.template dims<2>(); ++k)
            {
                //b(i,j,k) = h2 * f(I+i, J+j, K+k); //TODO
            }

    //--------------------------------------------------------------------------
    // Definition of placeholders. The order of them reflect the order the user
    // will deal with them especially the non-temporary ones, in the construction
    // of the domain
    typedef arg<0, storage_type > p_d_init;  //search direction
    typedef arg<1, storage_type > p_r_init;  //residual
    typedef arg<2, storage_type > p_b_init;  //rhs
    typedef arg<3, storage_type > p_Ax_init; //solution
    typedef arg<4, storage_type > p_x_init;  //solution
    typedef arg<5, parameter>     p_alpha_init; //step size

    // An array of placeholders to be passed to the domain
    typedef boost::mpl::vector<p_d_init,
                               p_r_init,
                               p_b_init,
                               p_Ax_init,
                               p_x_init,
                               p_alpha_init > accessor_list_init;

    typedef arg<0, storage_type > p_Ad_step0;  // A*d
    typedef arg<1, storage_type > p_d_step0;   //search direction

    typedef boost::mpl::vector<p_Ad_step0,
                               p_d_step0 > accessor_list_step0;

    typedef arg<0, storage_type > p_xNew_step1;  //solution
    typedef arg<1, storage_type > p_x_step1;     //solution
    typedef arg<2, storage_type > p_d_step1;     //search direction
    typedef arg<3, parameter >    p_alpha_step1; //step size

    typedef boost::mpl::vector<p_xNew_step1,
                               p_x_step1,
                               p_d_step1,
                               p_alpha_step1 > accessor_list_step1;

    typedef arg<0, storage_type > p_rNew_step2;  //residual t+1
    typedef arg<1, storage_type > p_r_step2;     //residual t
    typedef arg<2, storage_type > p_Ad_step2;    //A*d
    typedef arg<3, parameter >    p_alpha_step2; //step size

    typedef boost::mpl::vector<p_rNew_step2,
                               p_r_step2,
                               p_Ad_step2,
                               p_alpha_step2 > accessor_list_step2;

    typedef arg<0, storage_type > p_dNew_step3; //search direction t+1
    typedef arg<1, storage_type > p_rNew_step3; //residual t+1
    typedef arg<2, storage_type > p_d_step3;    //search direction t
    typedef arg<3, parameter >    p_beta_step3; //Gram-Schmidt ortog.

    typedef boost::mpl::vector<p_dNew_step3,
                               p_rNew_step3,
                               p_d_step3,
                               p_beta_step3 > accessor_list_step3;

   typedef arg<0, storage_type > p_out; //elementwise product a_t * b
   typedef arg<1, storage_type > p_a;
   typedef arg<2, storage_type > p_b;

   typedef boost::mpl::vector<p_out,
                              p_a,
                              p_b> accessor_list_alpha;

    /*
      Here we do lot of stuff
      1) We pass to the intermediate representation ::run function the description
         of the stencil, which is a multi-stage stencil (mss)
      2) The logical physical domain with the fields to use
      3) The actual domain dimensions
     */

    //start timer
    boost::timer::cpu_times lapse_time_run = {0,0,0};
    boost::timer::cpu_times lapse_time_d3point7 = {0,0,0};
    boost::timer::cpu_timer time;

    // construction of the domain for step phase
    gridtools::domain_type<accessor_list_init> domain_init
        (boost::fusion::make_vector(&d, &r, &b, &Ax, &x, &alpha));

    //instantiate stencil to perform initialization step of CG
    auto CG_init = gridtools::make_computation<gridtools::BACKEND>
        (
            domain_init, coords3d7pt,
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<d3point7>(p_Ax_init(), p_x_init()), // A * x, where x_0 = 0
                gridtools::make_esf<add_functor>(p_r_init(), p_b_init(), p_Ax_init(), p_alpha_init()), // r = b - Ax
                gridtools::make_esf<copy_functor>(p_d_init(), p_r_init()) // d = r
            )
        );

    //apply boundary conditions
    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = meta_.template get_halo_descriptor<0>();
    halos[1] = meta_.template get_halo_descriptor<1>();
    halos[2] = meta_.template get_halo_descriptor<2>();

    typename gridtools::boundary_apply
        <boundary_conditions<parallel_storage_info<metadata_t, partitioner_t>>, typename gridtools::bitmap_predicate>
        (halos,
         boundary_conditions<parallel_storage_info<metadata_t, partitioner_t>>(meta_, h),
         gridtools::bitmap_predicate(part.boundary())
        ).apply(x, d, xNew, dNew);

    // set addition parameter to -1 (subtraction): r = b + alpha A x
    double minus = -1;
    alpha.setValue(minus);

    //prepare and run single step of CG computation
    CG_init->ready();
    CG_init->steady();
    boost::timer::cpu_timer time_runInit;
    CG_init->run();
    lapse_time_run = lapse_time_run + time_runInit.elapsed();
    CG_init->finalize();

    //communicate halos
    std::vector<pointer_type::pointee_t*> vec(1);
    vec[0]=d.data().get();

    he.pack(vec);
    he.exchange();
    he.unpack(vec);

    MPI_Barrier(GCL_WORLD);

    double rTr_init; //initial residual
    double rTr_old; //used to remember value of global reduction between iterations

    /**
        Perform iterations of the CG
    */
    for(int iter=0; iter < TIME_STEPS; iter++) {

        // construction of the domains for the steps of CG
        gridtools::domain_type<accessor_list_step0> domain_step0
            (boost::fusion::make_vector(&Ad, ptr_d));

        gridtools::domain_type<accessor_list_step1> domain_step1
            (boost::fusion::make_vector(ptr_xNew, ptr_x, ptr_d, &alpha));

        gridtools::domain_type<accessor_list_step2> domain_step2
            (boost::fusion::make_vector(ptr_rNew, ptr_r, &Ad, &alpha));

        gridtools::domain_type<accessor_list_step3> domain_step3
            (boost::fusion::make_vector(ptr_dNew, ptr_rNew, ptr_d, &beta));

        gridtools::domain_type<accessor_list_alpha> domain_alpha_nominator
            (boost::fusion::make_vector(&tmp, ptr_r, ptr_r));

        gridtools::domain_type<accessor_list_alpha> domain_alpha_denominator
            (boost::fusion::make_vector(&tmp, ptr_d, &Ad));

        gridtools::domain_type<accessor_list_alpha> domain_beta_nominator
            (boost::fusion::make_vector(&tmp, ptr_rNew, ptr_rNew));


        //instantiate stencils to perform steps of CG
        auto CG_step0 = gridtools::make_computation<gridtools::BACKEND>
            (
                domain_step0, coords3d7pt,
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<d3point7>( p_Ad_step0(),
                                                   p_d_step0() ) // A * d
                )
            );


        auto CG_step1 = gridtools::make_computation<gridtools::BACKEND>
            (
                domain_step1, coords3d7pt,
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<add_functor>(p_xNew_step1(),
                                                     p_x_step1(),
                                                     p_d_step1(),
                                                     p_alpha_step1()) // x_(i+1) = x_i + alpha * d_i
                )
            );


        auto CG_step2 = gridtools::make_computation<gridtools::BACKEND>
            (
                domain_step2, coords3d7pt,
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<add_functor>(p_rNew_step2(),
                                                     p_r_step2(),
                                                     p_Ad_step2(),
                                                     p_alpha_step2()) // r_(i+1) = r_i - alpha * Ad_i
                )
            );

        auto CG_step3 = gridtools::make_computation<gridtools::BACKEND>
            (
                domain_step3, coords3d7pt,
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<add_functor>(p_dNew_step3(),
                                                     p_rNew_step3(),
                                                     p_d_step3(),
                                                     p_beta_step3()) // d_(i+1) = r_(i+1) + beta * d_i
                )
            );

        auto stencil_alpha_nom = make_computation< gridtools::BACKEND >
            (
                domain_alpha_nominator, coords3d7pt,
                gridtools::make_mss
                (
                    execute< forward >(),
                    make_esf<product_functor>(p_out(),
                                              p_a(),
                                              p_b()) // r_T * r
                ),
                make_reduction< reduction_functor, binop::sum >(0.0, p_out()) // sum(r_T * r)
            );

        auto stencil_alpha_denom = make_computation< gridtools::BACKEND >
            (
                domain_alpha_denominator, coords3d7pt,
                gridtools::make_mss
                (
                    execute< forward >(),
                    make_esf<product_functor>(p_out(),
                                              p_a(),
                                              p_b()) // d_T * A * d
                ),
                make_reduction< reduction_functor, binop::sum >(0.0, p_out()) // sum(d_T * A * d)
            );

            auto stencil_beta_nom = make_computation< gridtools::BACKEND >
            (
                domain_beta_nominator, coords3d7pt,
                gridtools::make_mss
                (
                    execute< forward >(),
                    make_esf<product_functor>(p_out(),
                                              p_a(),
                                              p_b()) // rNew_T * rNew
                ),
                make_reduction< reduction_functor, binop::sum >(0.0, p_out()) // sum(rNew_T * rNew)
            );

        boost::timer::cpu_timer time_iteration;

        // A * d
        CG_step0->ready();
        CG_step0->steady();
        boost::timer::cpu_timer time_run0;
        CG_step0->run();
        lapse_time_d3point7 = lapse_time_d3point7 + time_run0.elapsed();
        lapse_time_run = lapse_time_run + time_run0.elapsed();
        CG_step0->finalize();

        // compute step size alpha
        double rTr_global; //nominator
        double rTr;

        if (iter == 0)
        {
            stencil_alpha_nom->ready();
            stencil_alpha_nom->steady();
            boost::timer::cpu_timer time_alphaNom;
            rTr = stencil_alpha_nom->run(); // r_T * r (at time t)
            lapse_time_run = lapse_time_run + time_alphaNom.elapsed();
            stencil_alpha_nom->finalize();
            MPI_Allreduce(&rTr, &rTr_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            rTr_init = sqrt(rTr_global);
        }
        else
        {
            rTr_global = rTr_old; //reuse nominator from beta in previous time-step
        }

        stencil_alpha_denom->ready();
        stencil_alpha_denom->steady();
        boost::timer::cpu_timer time_alphaDenom;
        double dTAd = stencil_alpha_denom->run(); // d_T * A * d
        lapse_time_run = lapse_time_run + time_alphaDenom.elapsed();
        stencil_alpha_denom->finalize();
        double dTAd_global;
        MPI_Allreduce(&dTAd, &dTAd_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        alpha.setValue(rTr_global/dTAd_global);
        #ifdef DEBUG
        if (PID == 0) printf("Alpha = %f\n", rTr_global/dTAd_global);
        #endif

        // x_(i+1) = x_i + alpha * d_i
        CG_step1->ready();
        CG_step1->steady();
        boost::timer::cpu_timer time_run1;
        CG_step1->run();
        lapse_time_run = lapse_time_run + time_run1.elapsed();
        CG_step1->finalize();

        // r_(i+1) = r_i - alpha * Ad_i
        alpha.setValue(-1. * alpha.getValue());
        CG_step2->ready();
        CG_step2->steady();
        boost::timer::cpu_timer time_run2;
        CG_step2->run();
        lapse_time_run = lapse_time_run + time_run2.elapsed();
        CG_step2->finalize();

        //compute Gram–Schmidt orthogonalization parameter beta
        stencil_beta_nom->ready();
        stencil_beta_nom->steady();
        boost::timer::cpu_timer time_betaNom;
        double rTrnew = stencil_beta_nom->run(); // r_T * r (at time t+1) //TODO: reuse at next iteration in alpha
        lapse_time_run = lapse_time_run + time_betaNom.elapsed();
        stencil_beta_nom->finalize();
        double rTrnew_global;
        MPI_Allreduce(&rTrnew, &rTrnew_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        rTr_old = rTrnew_global; //save value to reuse it in alpha

        beta.setValue(rTrnew_global/rTr_global); // reusing r_T*r from computation of alpha
        #ifdef DEBUG
        if (PID == 0) printf("Beta = %f\n", rTrnew_global/rTr_global);
        #endif

        // d_(i+1) = r_(i+1) + beta * d_i
        CG_step3->ready();
        CG_step3->steady();
        boost::timer::cpu_timer time_run3;
        CG_step3->run();
        lapse_time_run = lapse_time_run + time_run3.elapsed();
        CG_step3->finalize();

        //communicate halos
        std::vector<pointer_type::pointee_t*> vec(2);
        vec[0]=d.data().get();
        vec[1]=dNew.data().get();

        he.pack(vec);
        he.exchange();
        he.unpack(vec);

        MPI_Barrier(GCL_WORLD);

        //swap input and output fields
        storage_type* swap;
        swap = ptr_x;
        ptr_x = ptr_xNew;
        ptr_xNew = swap;
        swap = ptr_r;
        ptr_r = ptr_rNew;
        ptr_rNew = swap;
        swap = ptr_d;
        ptr_d = ptr_dNew;
        ptr_dNew = swap;

        boost::timer::cpu_times lapse_time_iteration = time_iteration.elapsed();
        if (PID == 0)
        {
            std::cout << std::endl << "Iteration " << iter << ": [time]" << boost::timer::format(lapse_time_iteration);
            std::cout << "Iteration " << iter << ": [residual] " << sqrt(rTrnew_global)/rTr_init << std::endl;
        }

    }

    boost::timer::cpu_times lapse_time = time.elapsed();

    if (gridtools::PID == 0){
        std::cout << std::endl << "TOTAL TIME: " << boost::timer::format(lapse_time);
        std::cout << "TIME SPENT IN RUN STAGE:" << boost::timer::format(lapse_time_run);
        std::cout << "d3point7 MFLOPS: " << MFLOPS(10,d1,d2,d3,nt,lapse_time_d3point7.wall) << std::endl;
        std::cout << "d3point7 MLUPs: " << MLUPS(d1,d2,d3,nt,lapse_time_d3point7.wall) << std::endl << std::endl;
    }

#ifdef DEBUG
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "x" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        x.print(file);
    }
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "xNew" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        xNew.print(file);
    }
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "Ax" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        Ax.print(file);
    }
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "r" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        r.print(file);
    }
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "rNew" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        rNew.print(file);
    }
      
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "b" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        b.print(file);
    }
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "d" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        d.print(file);
    }
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "dNew" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        dNew.print(file);
    }

#endif

    gridtools::GCL_Finalize();

    return true;
    }
}