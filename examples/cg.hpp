#pragma once

//disable pedantic mode for the global accessor
#define PEDANTIC_DISABLED

#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/backend.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>

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

namespace cg{

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
    Parallel copy of one field done on the backend.
    @param out Destination domain.
    @param int Source domain.
*/
struct copy_functor {
    typedef const accessor<0, enumtype::inout> out;
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
        dom(out{}) = 7.0 * dom(in{})
                    - (dom(in{x(-1)})+dom(in{x(+1)}))
                    - (dom(in{y(-1)})+dom(in{y(+1)}))
                    - (dom(in{z(-1)})+dom(in{z(+1)}));
    }
};

/** @brief generic argument type
   Minimal interface to be passed as an argument to the user functor.
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
    template <typename Direction, typename DataField0, typename DataField1>
    GT_FUNCTION
    void operator()(Direction,
                    DataField0 & data_field0,
                    DataField1 & data_field1,
                    uint_t i, uint_t j, uint_t k) const {
        // get global indices on the boundary
        size_t I = m_partitioner.get_low_bound(0) + i;
        size_t J = m_partitioner.get_low_bound(1) + j;
        size_t K = m_partitioner.get_low_bound(2) + k;
        data_field0(i,j,k) = g(h*I, h*J, h*K);
        data_field1(i,j,k) = g(h*I, h*J, h*K);
    }
};
/*******************************************************************************/
/*******************************************************************************/

bool solver(uint_t xdim, uint_t ydim, uint_t zdim, uint_t nt) {

    gridtools::GCL_Init();

    // domain is encapsulated in boundary layer from both sides in each dimension
    // these are just inned domain dimension
    uint_t d1 = xdim;
    uint_t d2 = ydim;
    uint_t d3 = zdim;
    uint_t TIME_STEPS = nt;

    // force square domain
    if (!(xdim==ydim && ydim==zdim)) {
        printf("Please run with dimensions X=Y=Z\n");
        return false;
    }

    // step size, add +2 for boundary layer
    double h = 1./(xdim+2+1);
    double h2 = h*h;

    if(PID == 0){
        printf("Running for %d x %d x %d, %d iterations\n", xdim+2, ydim+2, zdim+2, nt);
        printf("Step size: %f\n", h);
    }

#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
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
    typedef storage_type::original_storage::pointer_type pointer_type;

    typedef gridtools::halo_exchange_dynamic_ut<layout_t,
                                                gridtools::layout_map<0, 1, 2>,
                                                pointer_type::pointee_t,
                                                MPI_3D_process_grid_t<3> ,
                                                gridtools::gcl_cpu,
                                                gridtools::version_manual> pattern_type;

    pattern_type he(pattern_type::grid_type::period_type(false, false, false), GCL_WORLD, &dimensions);

    // 7pt 3D stencil with symmetry distributed storage
    array<ushort_t, 3> padding{1,1,0};
    array<ushort_t, 3> halo{1,1,1};
    typedef partitioner_trivial<cell_topology<topology::cartesian<layout_map<0,1,2> > >, pattern_type::grid_type> partitioner_t;
    partitioner_t part(he.comm(), halo, padding);
    parallel_storage_info<metadata_t, partitioner_t> meta_(part, d1, d2, d3);
    auto metadata_=meta_.get_metadata();

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after.
    // Iteration space is defined within axis.
    gridtools::grid<axis, partitioner_t> coords3d7pt(part, meta_);

    //k dimension not partitioned
    coords3d7pt.value_list[0] = 1; //specifying index of the splitter<0,-1>
    coords3d7pt.value_list[1] = d3; //specifying index of the splitter<1,-1>

    //--------------------------------------------------------------------------
    // Definition of the actual data fields that are used for input/output

    storage_type b(metadata_, 0., "RHS vector");
    storage_type x(metadata_, 0., "Solution vector t");
    storage_type xNew(metadata_, 0., "Solution vector t+1");
    storage_type Ax(metadata_, 0., "Multiplied sol. vector Ax at time t");
    storage_type r(metadata_, 0., "Residual t");
    storage_type rNew(metadata_, 0., "Residual t+1");
    storage_type d(metadata_, 0., "Direction vector t");
    storage_type dNew(metadata_, 0., "Direction vector t+1");
    storage_type Ad(metadata_, 0., "Multiplied direction vector");
    //storage_type *ptr_in7pt = &in7pt, *ptr_out7pt = &out7pt;

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

    // initialize the local domain
    for(uint_t i=0; i<metadata_.template dims<0>(); ++i)
        for(uint_t j=0; j<metadata_.template dims<1>(); ++j)
            for(uint_t k=0; k<metadata_.template dims<2>(); ++k)
            {
                b(i,j,k) = h2 * f(I+i, J+j, K+k);
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

    /*
      Here we do lot of stuff
      1) We pass to the intermediate representation ::run function the description
      of the stencil, which is a multi-stage stencil (mss)
      2) The logical physical domain with the fields to use
      3) The actual domain dimensions
     */


    //start timer
    boost::timer::cpu_times lapse_time_run = {0,0,0};
    boost::timer::cpu_timer time;

    // construction of the domain for step phase
    gridtools::domain_type<accessor_list_init> domain_init
        (boost::fusion::make_vector(&d, &r, &b, &Ax, &x, &alpha));

    //instantiate stencil to perform initialization step of CG
    #ifdef __CUDACC__
        gridtools::computation* stencil_init =
    #else
            boost::shared_ptr<gridtools::computation> stencil_init =
    #endif
          gridtools::make_computation<gridtools::BACKEND>
            (
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<d3point7>(p_Ax_init(), p_x_init()), // Ax
                    gridtools::make_esf<add_functor>(p_r_init(), p_b_init(), p_Ax_init(), p_alpha_init()), // r = b - Ax
                    gridtools::make_esf<copy_functor>(p_d_init(), p_r_init()) // d = r
                ),
                domain_init, coords3d7pt
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
        ).apply(x, d);

    // set addition param to substraction: c = b - a
    double minus = -1;
    alpha.setValue(minus);

    //prepare and run single step of stencil computation
    stencil_init->ready();
    stencil_init->steady();
    boost::timer::cpu_timer time_run;
    stencil_init->run();
    lapse_time_run = lapse_time_run + time_run.elapsed();
    stencil_init->finalize();

    //communicate halos
    std::vector<pointer_type::pointee_t*> vec(2);
    vec[0]=x.data().get();
    vec[1]=d.data().get();

    he.pack(vec);
    he.exchange();
    he.unpack(vec);

    MPI_Barrier(GCL_WORLD);

    /** 
        Perform iterations of the CG
    */
    for(int i=0; i < TIME_STEPS; i++) {

        //compute step size alpha
        double ddot = 1;
        alpha.setValue(ddot);

        // construction of the domain for the step of CG
        gridtools::domain_type<accessor_list_step1> domain_step1
            (boost::fusion::make_vector(&xNew, &x, &d, &alpha));

        gridtools::domain_type<accessor_list_step2> domain_step2
            (boost::fusion::make_vector(&rNew, &r, &Ad, &alpha));

        gridtools::domain_type<accessor_list_step3> domain_step3
            (boost::fusion::make_vector(&dNew, &rNew, &d, &beta));

        //instantiate stencil to perform step of CG
        #ifdef __CUDACC__
            gridtools::computation* stencil_step1 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step1 =
        #endif
              gridtools::make_computation<gridtools::BACKEND>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<add_functor>(p_xNew_step1(),
                                                         p_x_step1(),
                                                         p_d_step1(),
                                                         p_alpha_step1()) // x_(i+1) = x_i + alpha * d_i
                    ),
                    domain_step1, coords3d7pt
                );

        #ifdef __CUDACC__
            gridtools::computation* stencil_step2 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step2 =
        #endif
              gridtools::make_computation<gridtools::BACKEND>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<add_functor>(p_rNew_step2(),
                                                         p_r_step2(),
                                                         p_Ad_step2(),
                                                         p_alpha_step2()) // r_(i+1) = r_i + alpha * Ad_i
                    ),
                    domain_step2, coords3d7pt
                );

        #ifdef __CUDACC__
            gridtools::computation* stencil_step3 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step3 =
        #endif
              gridtools::make_computation<gridtools::BACKEND>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<add_functor>(p_dNew_step3(),
                                                         p_rNew_step3(),
                                                         p_d_step3(),
                                                         p_beta_step3()) // d_(i+1) = r_(i+1) + beta * d_i
                    ),
                    domain_step3, coords3d7pt
                );



        //prepare and run steps of stencil computation
        stencil_step1->ready();
        stencil_step1->steady();
        boost::timer::cpu_timer time_run1;
        stencil_step1->run();
        lapse_time_run = lapse_time_run + time_run1.elapsed();
        stencil_step1->finalize();

        stencil_step2->ready();
        stencil_step2->steady();
        boost::timer::cpu_timer time_run2;
        stencil_step2->run();
        lapse_time_run = lapse_time_run + time_run2.elapsed();
        stencil_step2->finalize();

        //compute Gram–Schmidt orthogonalization parameter beta
        double orto = 1.;
        beta.setValue(orto);

        stencil_step3->ready();
        stencil_step3->steady();
        boost::timer::cpu_timer time_run3;
        stencil_step3->run();
        lapse_time_run = lapse_time_run + time_run3.elapsed();
        stencil_step3->finalize();

        //swap input and output fields
        //storage_type* tmp = ptr_out7pt;
        //ptr_out7pt = ptr_in7pt;
        //ptr_in7pt = tmp;
    }

    boost::timer::cpu_times lapse_time = time.elapsed();

    if(gridtools::PID == 0){
        std::cout << "TIME d3point7 TOTAL: " << boost::timer::format(lapse_time);
        std::cout << "TIME d3point7 RUN:" << boost::timer::format(lapse_time_run);
        std::cout << "TIME d3point7 MFLOPS: " << MFLOPS(10,d1,d2,d3,nt,lapse_time_run.wall) << std::endl;
        std::cout << "TIME d3point7 MLUPs: " << MLUPS(d1,d2,d3,nt,lapse_time_run.wall) << std::endl << std::endl;
    }

#ifndef NDEBUG1
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
#endif

    gridtools::GCL_Finalize();

    return true;
    }
}
