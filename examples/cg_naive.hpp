#pragma once

//disable pedantic mode for the global accessor
#define PEDANTIC_DISABLED

#define REL_TOL
#define MY_VERBOSE
#define a_i 0.00001

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
#include "Timers.hpp"
#include "cg.h"

//time t is in ns, returns MFLOPS
inline double MFLOPS(int numops, double X, double Y, double Z, double NT, double t) { return numops*X*Y*Z*NT*1000.0/t; }
inline double MLUPS(double X, double Y, double Z, double NT, double t) { return X*Y*Z*NT*1000.0/t; }

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
        typedef accessor<1, enumtype::in, extent<-1,1,-1,1,-1,1> > in; // this says to access 6 neighbors
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
                    // Get global indices on the boundary
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

    bool solver(uint_t xdim, uint_t ydim, uint_t zdim, uint_t MAX_ITER, const double EPS, Timers *timers) {

        // Initialize MPI
        //gridtools::GCL_Init();

        // Domain is encapsulated in boundary layer from both sides in each dimension,
        // Boundary is added as extra layer to each dimension 
        uint_t d1 = xdim;
        uint_t d2 = ydim;
        uint_t d3 = zdim;

        // Enforce square domain
        // if (!(xdim==ydim && ydim==zdim)) {
        //     if (PID==0) printf("Please run with dimensions X=Y=Z\n");
        //     return false;
        // }

        // Step size, add +2 for boundary layer
        double h = 1./(d1+2+1);
        double h2 = h*h;

        if (PID == 0){
            printf("Running CG for domain %d x %d x %d, %d iterations, tolerance %.4f\n", xdim, ydim, zdim, MAX_ITER, EPS);
        }

#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, GRIDBACKEND, Block >
#else
#define BACKEND backend<Host, GRIDBACKEND, Naive >
#endif

        // Start timer
        boost::timer::cpu_times lapse_time_run = {0,0,0};
        boost::timer::cpu_times lapse_time_d3point7 = {0,0,0};
        boost::timer::cpu_timer time;

        timers->start(Timers::TIMER_GLOBAL);

        // Create processor grid
        array<int, 3> dimensions{0,0,0};
        MPI_3D_process_grid_t<3>::dims_create(PROCS, 2, dimensions);
        dimensions[2]=1;

        // 2D partitioning scheme info
        int xprocs = dimensions[0];
        int yprocs = dimensions[1];

        // Split the communicator based on the color and use the
        // original rank for ordering
        MPI_Comm xdim_comm;
        MPI_Comm_split(MPI_COMM_WORLD, PID % yprocs, PID, &xdim_comm);

        // Prepare types for the data storage
        //                   strides  1 x xy
        //                      dims  x y z
        typedef gridtools::layout_map<2,1,0> layout_t;
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
        array<ushort_t, 3> padding{1,1,1}; // global halo, 1-Dirichlet, 0-Neumann
        array<ushort_t, 3> halo{1,1,1}; // number of layers to communicate to neighboring processes
        typedef partitioner_trivial<cell_topology<topology::cartesian<layout_map<2,1,0> > >, pattern_type::grid_type> partitioner_t;
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

        storage_type b    (metadata_, 0., "RHS vector");
        storage_type x    (metadata_, 0., "Solution vector t");
        storage_type xNew (metadata_, 0., "Solution vector t+1");
        storage_type Ax   (metadata_, 0., "Vector Ax at time t");
        storage_type BCx  (metadata_, 0., "BC applied to vector x");
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
        parameter beta;  //orthogonalization parameter

        // Set up halo
        he.add_halo<0>(meta_.template get_halo_gcl<0>());
        he.add_halo<1>(meta_.template get_halo_gcl<1>());
        he.add_halo<2>(meta_.template get_halo_gcl<2>());
        he.setup(2);

        // Get global offsets
        size_t I = meta_.get_low_bound(0);
        size_t J = meta_.get_low_bound(1);
        size_t K = meta_.get_low_bound(2);

        // Partitioning info
        //std::cout << "I #" << PID << ": " << meta_.get_low_bound(0) << " - " << meta_.get_up_bound(0) << std::endl;
        //std::cout << "J #" << PID << ": " << meta_.get_low_bound(1) << " - " << meta_.get_up_bound(1) << std::endl;

        // Initialize the local RHS vector domain
        std::srand(std::time(0));
        std::srand(42);
        for (uint_t i=1; i<metadata_.template dims<0>() - 1 ; ++i)
            for (uint_t j=1; j<metadata_.template dims<1>() - 1; ++j)
                for (uint_t k=1; k<metadata_.template dims<2>() -1 ; ++k)
                {
                    b(i,j,k) = (std::rand()/(double)RAND_MAX > 0.5 ? 1.0 : -1.0);
                    //x(i,j,k) = 100*(i+I) + 10*(j+J) + k;
                    //b(i,j,k) = 100*(i+I) + 10*(j+J) + k;
                }

        //--------------------------------------------------------------------------
        // Definition of placeholders. The order of them reflect the order the user
        // will deal with them especially the non-temporary ones, in the construction
        // of the domain

        typedef arg<0, storage_type > p_tmp_init;//residual norm
        typedef arg<1, storage_type > p_d_init;  //search direction
        typedef arg<2, storage_type > p_r_init;  //residual
        typedef arg<3, storage_type > p_b_init;  //rhs
        typedef arg<4, storage_type > p_Ax_init; //solution
        typedef arg<5, storage_type > p_BCx_init; //solution
        typedef arg<6, storage_type > p_x_init; //solution
        typedef arg<7, parameter>     p_alpha_init; //step size
        typedef arg<8, parameter>     p_beta_init; //step size

        // An array of placeholders to be passed to the domain
        typedef boost::mpl::vector<p_tmp_init,
                p_d_init,
                p_r_init,
                p_b_init,
                p_Ax_init,
                p_BCx_init,
                p_x_init,
                p_alpha_init,
                p_beta_init > accessor_list_init;

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

        typedef arg<0, storage_type > p_tmp_step2;   //residual norm
        typedef arg<1, storage_type > p_rNew_step2;  //residual t+1
        typedef arg<2, storage_type > p_r_step2;     //residual t
        typedef arg<3, storage_type > p_Ad_step2;    //A*d
        typedef arg<4, parameter >    p_alpha_step2; //step size

        typedef boost::mpl::vector<p_tmp_step2,
                p_rNew_step2,
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

        // variables for dot product reductions
        double rTr;
        double rTr_global;

        double rTrnew;
        double rTrnew_global;

        double dTAd; 
        double dTAd_global;

        double rTr_init; //initial residual, sqrt(rTr_global)

        /*
           Here we do lot of stuff
           1) We pass to the intermediate representation ::run function the description
           of the stencil, which is a multi-stage stencil (mss)
           2) The logical physical domain with the fields to use
           3) The actual domain dimensions
           */

        // Construction of the domain for step phase
        gridtools::domain_type<accessor_list_init> domain_init
            (boost::fusion::make_vector(&tmp, &d, &r, &b, &Ax, &BCx, &x, &alpha, &beta));

        // Instantiate stencil to perform initialization step of CG
        auto CG_init = gridtools::make_computation<gridtools::BACKEND>
            (
             domain_init, coords3d7pt,
             gridtools::make_mss // mss_descriptor
             (
              execute<forward>(),
              gridtools::make_esf<d3point7>(p_Ax_init(), p_x_init()), // A * x, where x_0 = 0
              gridtools::make_esf<add_functor>(p_Ax_init(), p_Ax_init(), p_BCx_init(), p_alpha_init()), // Ax = Ax_0 + BCx_0
              gridtools::make_esf<add_functor>(p_r_init(), p_b_init(), p_Ax_init(), p_beta_init()), // r = b - Ax
              gridtools::make_esf<copy_functor>(p_d_init(), p_r_init()), // d = r
              gridtools::make_esf<product_functor>(p_tmp_init(), p_r_init(), p_r_init()) // r' .* r
             ),
             make_reduction< reduction_functor, binop::sum >(0.0, p_tmp_init()) // sum(r'.*r)
            );

        // Apply boundary conditions
        // gridtools::array<gridtools::halo_descriptor, 3> halos;
        // halos[0] = meta_.template get_halo_descriptor<0>();
        // halos[1] = meta_.template get_halo_descriptor<1>();
        // halos[2] = meta_.template get_halo_descriptor<2>();

        // typename gridtools::boundary_apply
        //     <boundary_conditions<parallel_storage_info<metadata_t, partitioner_t>>, typename gridtools::bitmap_predicate>
        //     (halos,
        //      boundary_conditions<parallel_storage_info<metadata_t, partitioner_t>>(meta_, h),
        //      gridtools::bitmap_predicate(part.boundary())
        //     ).apply(x, d, xNew, dNew);


        // Set addition parameter to -1 (subtraction): r = b + alpha A x
        double minus = -1;
        double plus = 1;
        alpha.setValue(plus);
        beta.setValue(minus);

        //communicate halos - if x is set to zeros this is not necessary
        std::vector<pointer_type::pointee_t*> vec(1);
        vec[0]=x.data().get();

        he.pack(vec);
        he.exchange();
        he.unpack(vec);

        //do the initial phase of CG
        CG_init->ready();
        CG_init->steady();
        boost::timer::cpu_timer time_runInit;
        rTr = CG_init->run();
        lapse_time_run = lapse_time_run + time_runInit.elapsed();
        CG_init->finalize();
        MPI_Allreduce(&rTr, &rTr_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        rTr_init = sqrt(rTr_global); //initial residual



#ifndef MY_VERBOSE
        if (PID == 0) {
#ifdef REL_TOL
            std::cout << "Iteration 0: [residual] " << sqrt(rTr_global)/rTr_init << std::endl << std::endl;
#else
            std::cout << "Iteration 0: [residual] " << rTr_init << std::endl << std::endl;
#endif
        }
#endif

        //communicate halos
        vec[0]=d.data().get();

        he.pack(vec);
        he.exchange();
        he.unpack(vec);

        MPI_Barrier(GCL_WORLD);

        /**
          Perform iterations of the CG
          */

        int iter;
        double residual;

        // get local dims of the domain
        int xdim_local = metadata_.template dims<0>() - 2;
        int ydim_local = metadata_.template dims<1>() - 2;
        int zdim_local = metadata_.template dims<2>() - 2;

        // allocate edge for BC
        double *xedge = (double *)malloc(sizeof(double) * xdim);

        for(iter=1; iter <= MAX_ITER; iter++) {

            // construction of the domains for the steps of CG
            gridtools::domain_type<accessor_list_step0> domain_step0
                (boost::fusion::make_vector(&Ad, ptr_d));

            gridtools::domain_type<accessor_list_step1> domain_step1
                (boost::fusion::make_vector(ptr_xNew, ptr_x, ptr_d, &alpha));

            gridtools::domain_type<accessor_list_step2> domain_step2
                (boost::fusion::make_vector(&tmp, ptr_rNew, ptr_r, &Ad, &alpha));

            gridtools::domain_type<accessor_list_step3> domain_step3
                (boost::fusion::make_vector(ptr_dNew, ptr_rNew, ptr_d, &beta));

            gridtools::domain_type<accessor_list_alpha> domain_alpha_nominator
                (boost::fusion::make_vector(&tmp, ptr_r, ptr_r));

            gridtools::domain_type<accessor_list_alpha> domain_alpha_denominator
                (boost::fusion::make_vector(&tmp, ptr_d, &Ad));

            gridtools::domain_type<accessor_list_alpha> domain_beta_nominator
                (boost::fusion::make_vector(&tmp, ptr_rNew, ptr_rNew));


            // Instantiate stencils to perform steps of CG
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
                      p_alpha_step2()), // r_(i+1) = r_i - alpha * Ad_i
                  make_esf<product_functor>(p_tmp_step2(), p_rNew_step2(), p_rNew_step2()) // r' .* r
                 ),
                 make_reduction< reduction_functor, binop::sum >(0.0, p_tmp_step2()) // sum(r'.*r)
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

            boost::timer::cpu_timer time_iteration;

            // A * d
            CG_step0->ready();
            CG_step0->steady();
            boost::timer::cpu_timer time_run0;
            timers->start(Timers::TIMER_COMPUTE_STENCIL_INNER);
            CG_step0->run();
            timers->stop(Timers::TIMER_COMPUTE_STENCIL_INNER);
            lapse_time_d3point7 = lapse_time_d3point7 + time_run0.elapsed();
            lapse_time_run = lapse_time_run + time_run0.elapsed();
            CG_step0->finalize();

            // if(PID==0) { printf("d\n"); ptr_d->print(); }
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==1) {  printf("d\n"); ptr_d->print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==2) {  printf("d\n"); ptr_d->print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==3) {  printf("d\n"); ptr_d->print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==4) {  printf("d\n"); ptr_d->print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==5) {  printf("d\n"); ptr_d->print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==6) {  printf("d\n"); ptr_d->print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==7) {  printf("d\n"); ptr_d->print();}

            // if(PID==0) {printf("Ad\n"); Ad.print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==1)  {printf("Ad\n"); Ad.print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==2)  {printf("Ad\n"); Ad.print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==3)  {printf("Ad\n"); Ad.print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==4)  {printf("Ad\n"); Ad.print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==5)  {printf("Ad\n"); Ad.print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==6)  {printf("Ad\n"); Ad.print();}
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(PID==7) { printf("Ad\n"); Ad.print();}
           
            /** Apply BC to domain d
             *   1. allreduce the first and last edge of the domain in x direction
             *   2. perform dense mat-vec that represents addition of BC
             *   3. add the BC to the domain Ax
             */
            timers->start(Timers::TIMER_COMPUTE_STENCIL_BORDER);

            //get pointer to the domain X data
            double *d_data = ptr_d->data().get();

            // apply BC to the  (x,y=0,z=0) edge of the domain
            if (PID % yprocs == 0)
            {
                //get index [1,1,0] to skip halo/BC of the local domain
                int start = (xdim_local+2)*(ydim_local+2) + (xdim_local+2) + 1;

                // Gather all partial averages down to all the processes
                MPI_Allgather(&d_data[start], xdim_local, MPI_DOUBLE, xedge, xdim_local, MPI_DOUBLE,  xdim_comm);

                double *Ad_data = &(Ad.data().get())[start]; 
                
                double tmp = 0.0;
                for (int col = 0; col < xdim; col++)
                {
                    //BC * x
                    tmp += a_i * xedge[col]; 
                }
                
                for (int row = 0; row < xdim_local; row++)
                {
                    //add BD to the stencil
                    Ad_data[row] += tmp; 
                }

            }

            // apply BC to the edge (x,y=N,z=N)
            if (PID % yprocs == yprocs - 1)
            {
                int start = ((xdim_local + 2 ) * (ydim_local + 2) * (zdim_local )) + (xdim_local + 2) * (ydim_local ) + 1;

                // Gather all partial averages down to all the processes
                MPI_Allgather(&d_data[start], xdim_local, MPI_DOUBLE, xedge, xdim_local, MPI_DOUBLE,  xdim_comm);

                double *Ad_data = &(Ad.data().get())[start]; 
                
                double tmp = 0.0;
                for (int col = 0; col < xdim; col++)
                {
                    //BC * x
                    tmp += a_i * xedge[col]; 
                }
                
                for (int row = 0; row < xdim_local; row++)
                {
                    //add BD to the stencil
                    Ad_data[row] += tmp; 
                }

            }

            //  if(PID==0) {printf("Ad\n"); Ad.print();}
            //  MPI_Barrier(MPI_COMM_WORLD);
            //  if(PID==1)  {printf("Ad\n"); Ad.print();}
            //  MPI_Barrier(MPI_COMM_WORLD);
            //  if(PID==2)  {printf("Ad\n"); Ad.print();}
            //  MPI_Barrier(MPI_COMM_WORLD);
            //  if(PID==3)  {printf("Ad\n"); Ad.print();}
            //  MPI_Barrier(MPI_COMM_WORLD);
            //  if(PID==4)  {printf("Ad\n"); Ad.print();}
            //  MPI_Barrier(MPI_COMM_WORLD);
            //  if(PID==5)  {printf("Ad\n"); Ad.print();}
            //  MPI_Barrier(MPI_COMM_WORLD);
            //  if(PID==6)  {printf("Ad\n"); Ad.print();}
            //  MPI_Barrier(MPI_COMM_WORLD);
            //  if(PID==7) { printf("Ad\n"); Ad.print();}

            timers->stop(Timers::TIMER_COMPUTE_STENCIL_BORDER);

            // Denominator of alpha
            stencil_alpha_denom->ready();
            stencil_alpha_denom->steady();
            boost::timer::cpu_timer time_alphaDenom;
            dTAd = stencil_alpha_denom->run(); // d_T * A * d
            lapse_time_run = lapse_time_run + time_alphaDenom.elapsed();
            stencil_alpha_denom->finalize();
            MPI_Allreduce(&dTAd, &dTAd_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
            timers->start(Timers::TIMER_COMPUTE_DOTPROD); //measure r_i - a*Ad and r*'r and reduction
            rTrnew = CG_step2->run();
            timers->stop(Timers::TIMER_COMPUTE_DOTPROD);
            lapse_time_run = lapse_time_run + time_run2.elapsed();
            CG_step2->finalize();
            MPI_Allreduce(&rTrnew, &rTrnew_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            //compute Gram–Schmidt orthogonalization parameter beta
            beta.setValue(rTrnew_global/rTr_global); // reusing r_T*r from computation of alpha
#ifdef DEBUG
            if (PID == 0) printf("Beta = %f\n", rTrnew_global/rTr_global);
#endif

            rTr_global = rTrnew_global; //reuse nominator from beta in next time-step

            // d_(i+1) = r_(i+1) + beta * d_i
            CG_step3->ready();
            CG_step3->steady();
            boost::timer::cpu_timer time_run3;
            CG_step3->run();
            lapse_time_run = lapse_time_run + time_run3.elapsed();
            CG_step3->finalize();

            // Communicate halos
            std::vector<pointer_type::pointee_t*> vec(2);
            vec[0]=d.data().get();
            vec[1]=dNew.data().get();

            timers->start(Timers::TIMER_HALO_PACK);
            he.pack(vec);
            timers->stop(Timers::TIMER_HALO_PACK);

            timers->start(Timers::TIMER_HALO_ISEND_IRECV);
            he.exchange();
            timers->stop(Timers::TIMER_HALO_ISEND_IRECV);

            //timer UNPACK_WAIT is used to measure only unpack
            timers->start(Timers::TIMER_HALO_UNPACK_WAIT);
            he.unpack(vec);
            timers->stop(Timers::TIMER_HALO_UNPACK_WAIT);

            MPI_Barrier(GCL_WORLD);

            // Swap input and output fields
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

#ifdef REL_TOL
            residual =  sqrt(rTrnew_global)/rTr_init;
#else
            residual =  sqrt(rTrnew_global);
#endif

#ifndef MY_VERBOSE
            if (PID == 0)
            {
                std::cout << "Iteration " << iter << ": [time] " << boost::timer::format(lapse_time_iteration,8,"%w") << std::endl;
                std::cout << "Iteration " << iter << ": [residual] " << residual << std::endl << std::endl;
            }
#endif

            // Convergence test
            if (residual < EPS)
                break;
        }

        boost::timer::cpu_times lapse_time = time.elapsed();
        timers->stop(Timers::TIMER_GLOBAL);

#ifdef MY_VERBOSE
        if (PID == 0) {
            std::cout << "Total iteration count: " << iter-1 << std::endl;
            std::cout << "Residual: " << residual << std::endl;
        }
#endif

#ifndef MY_VERBOSE
        if (PID == 0) {
            std::cout << std::endl << "Total time: " << boost::timer::format(lapse_time,8,"%w") << std::endl;
            std::cout << "Total time in run stage: " << boost::timer::format(lapse_time_run,8,"%w") << std::endl;
            std::cout << "Stencil performance MFLOPS: " << MFLOPS(7,d1,d2,d3,MAX_ITER,lapse_time_d3point7.wall) << std::endl;
            std::cout << "Stencil performance MLUPs: " << MLUPS(d1,d2,d3,MAX_ITER,lapse_time_d3point7.wall) << std::endl << std::endl;
        }
#endif

        //gridtools::GCL_Finalize();
        MPI_Comm_free(&xdim_comm);
        free(xedge);
        return true;
    }
}
