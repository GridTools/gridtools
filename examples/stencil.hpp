#pragma once

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
inline double MFLOPS(int numops, double X, double Y, double Z, double NT, double t) { return numops*X*Y*Z*NT*1000.0/t; }
inline double MLUPS(double X, double Y, double Z, double NT, double t) { return X*Y*Z*NT*1000.0/t; }

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

namespace stencil__{

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
    template <typename Direction, typename DataField0>
    GT_FUNCTION
    void operator()(Direction,
                    DataField0 & data_field0,
                    uint_t i, uint_t j, uint_t k) const {
        // Get global indices on the boundary
        size_t I = m_partitioner.get_low_bound(0) + i;
        size_t J = m_partitioner.get_low_bound(1) + j;
        size_t K = m_partitioner.get_low_bound(2) + k;
        data_field0(i,j,k) = 0;//g(h*I, h*J, h*K);
    }
};
/*******************************************************************************/
/*******************************************************************************/

bool solver(uint_t xdim, uint_t ydim, uint_t zdim, uint_t nt) {

    // Initialize MPI
    gridtools::GCL_Init();

    // Domain is encapsulated in boundary layer from both sides in each dimension,
    // Boundary is added as extra layer to each dimension
    uint_t d1 = xdim;
    uint_t d2 = ydim;
    uint_t d3 = zdim; //TODO boundary layer
    uint_t TIME_STEPS = nt;

    // Enforce square domain
    // if (!(xdim==ydim && ydim==zdim)) {
    //     if (PID==0) printf("Please run with dimensions X=Y=Z\n");
    //     return false;
    // }

    // Step size, add +2 for boundary layer
    double h = 1./(d1+2+1);//TODO boundary layer
    double h2 = h*h;

    if (PID == 0){
        printf("Running for %d x %d x %d, %d iterations\n", xdim, ydim, zdim, nt);
        // printf("Step size: %f\n", h);
    }

#ifdef CUDA_EXAMPLE
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, GRIDBACKEND, Block >
#else
#define BACKEND backend<Host, GRIDBACKEND, Naive >
#endif
#endif

    //--------------------------------------------------------------------------
    // Create processor grid
    array<int, 3> dimensions{0,0,0};
    MPI_3D_process_grid_t<3>::dims_create(PROCS, 2, dimensions);
    dimensions[2]=1;

    // Prepare types for the data storage
    //                   strides  1 x xy
    //                      dims  x y z
    #ifdef CUDA_EXAMPLE
    typedef gridtools::layout_map<2,1,0> layout_t;
    #else
    typedef gridtools::layout_map<0,1,2> layout_t;
    #endif
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
    auto const& metadata_=meta_.get_metadata();

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

    storage_type x     (metadata_, 1., "Solution vector t");
    storage_type Ax    (metadata_, 0., "Vector Ax at time t");

    // Pointers to data-fields are swapped at each time-iteration
    storage_type *ptr_x = &x, *ptr_xNew = &Ax;

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

    // Initialize the RHS vector domain
    // for (uint_t i=0; i<metadata_.template dims<0>(); ++i)
    //     for (uint_t j=0; j<metadata_.template dims<1>(); ++j)
    //         for (uint_t k=0; k<metadata_.template dims<2>(); ++k)
    //         {
    //             //x(i,j,k) = 10000*(I+i) +  100*(J+j) + K+k;
    //         }

    //--------------------------------------------------------------------------
    // Definition of placeholders. The order of them reflect the order the user
    // will deal with them especially the non-temporary ones, in the construction
    // of the domain
    typedef arg<0, storage_type > p_Ax;  //residual
    typedef arg<1, storage_type > p_x;  //rhs

    // An array of placeholders to be passed to the domain
    typedef boost::mpl::vector<p_Ax,
                               p_x > accessor_list;

    /*
      Here we do lot of stuff
      1) We pass to the intermediate representation ::run function the description
         of the stencil, which is a multi-stage stencil (mss)
      2) The logical physical domain with the fields to use
      3) The actual domain dimensions
     */

    // Start timer
    boost::timer::cpu_times lapse_time_ReadySteady = {0,0,0};
    boost::timer::cpu_times lapse_time_run = {0,0,0};
    boost::timer::cpu_times lapse_time_finalize = {0,0,0};
    boost::timer::cpu_timer time;

    /**
        Perform iterations of the stencil
    */

    for (int iter=0; iter < TIME_STEPS; iter++) {

        // Construction of the domain for step phase
        gridtools::domain_type<accessor_list> domain
            (boost::fusion::make_vector(ptr_xNew, ptr_x));

        // Instantiate stencil to perform initialization step of CG
        #ifdef CXX11_ENABLED
            auto
        #else
        #ifdef __CUDACC__
            stencil*
        #else
            boost::shared_ptr<gridtools::stencil>
        #endif
        #endif
        stencil = gridtools::make_computation<gridtools::BACKEND>
            (
                domain, coords3d7pt,
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<d3point7>(p_Ax(), p_x()) // A * x, where x_0 = 1
                )
            );

        // Apply boundary conditions
        gridtools::array<gridtools::halo_descriptor, 3> halos;
        halos[0] = meta_.template get_halo_descriptor<0>();
        halos[1] = meta_.template get_halo_descriptor<1>();
        halos[2] = meta_.template get_halo_descriptor<2>();

        typename gridtools::boundary_apply
            <boundary_conditions<parallel_storage_info<metadata_t, partitioner_t>>, typename gridtools::bitmap_predicate>
            (halos,
             boundary_conditions<parallel_storage_info<metadata_t, partitioner_t>>(meta_, h),
             gridtools::bitmap_predicate(part.boundary())
            ).apply(*ptr_x);

        // Prepare and run single step of CG computation
        boost::timer::cpu_timer time_ReadySteady;
        stencil->ready();
        stencil->steady();
        lapse_time_ReadySteady = time_ReadySteady.elapsed();
        //std::cout << "TIME SPENT IN Ready/Steady:" << boost::timer::format(lapse_time_ReadySteady);

        boost::timer::cpu_timer time_run;
        stencil->run();
        lapse_time_run = lapse_time_run + time_run.elapsed();
        //std::cout << "TIME SPENT IN Run:" << boost::timer::format(time_run.elapsed());

        boost::timer::cpu_timer time_finalize;
        stencil->finalize();
        lapse_time_finalize = time_finalize.elapsed();
        //std::cout << "TIME SPENT IN Finalize:" << boost::timer::format(lapse_time_finalize);

        // #ifdef __CUDACC__
        // out.d2h_update();
        // in.d2h_update();
        // #endif

        //communicate halos //TODO - what about halo exchange before first computation? is it done automatically by partitioner?
        std::vector<pointer_type::pointee_t*> vec(1);
        vec[0]=ptr_xNew->data().get();

        he.pack(vec);
        he.exchange();
        he.unpack(vec);

        MPI_Barrier(GCL_WORLD);

        // Swap input and output fields
        storage_type* swap;
        swap = ptr_x;
        ptr_x = ptr_xNew;
        ptr_xNew = swap;

    } //end for

    boost::timer::cpu_times lapse_time = time.elapsed();

    if (gridtools::PID == 0){
        //std::cout << std::endl << "TOTAL TIME: " << boost::timer::format(lapse_time);
        //std::cout << "TIME SPENT IN RUN STAGE:" << boost::timer::format(lapse_time_run);
        std::cout << "d3point7 MFLOPS: " << MFLOPS(7,d1,d2,d3,TIME_STEPS,lapse_time_run.wall) << std::endl; //TODO: multiple processes??
        std::cout << "d3point7 MLUPs: " << MLUPS(d1,d2,d3,TIME_STEPS,lapse_time_run.wall) << std::endl << std::endl;

        boost::timer::cpu_times total_time = lapse_time_run + lapse_time_finalize + lapse_time_ReadySteady;
        std::cout << "d3point7+comm MFLOPS: " << MFLOPS(7,d1,d2,d3,TIME_STEPS,total_time.wall) << std::endl; //TODO: multiple processes??
        std::cout << "d3point7+comm MLUPs: " << MLUPS(d1,d2,d3,TIME_STEPS,total_time.wall) << std::endl << std::endl;
    }

#ifdef DEBUG
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "x" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        ptr_x->print(file);
    }
    {
        std::stringstream ss;
        ss << PID;
        std::string filename = "xNew" + ss.str() + ".txt";
        std::ofstream file(filename.c_str());
        ptr_xNew->print(file);
    }

#endif

    gridtools::GCL_Finalize();

    return true;
    }
}
