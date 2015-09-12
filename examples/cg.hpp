#pragma once

#include <gridtools.hpp>

#include <stencil-composition/backend.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>

#include <boost/timer/timer.hpp>
#include "cg.h"

/*
  @file This file shows an implementation of the various stencil operations.

 1nd order in time:
  3-point constant-coefficient stencil in one dimension, with symmetry.
  7-point constant-coefficient isotropic stencil in three dimensions, with symmetry.
  7-point variable-coefficient stencil in three dimension, with no coefficient symmetry.
  25-point constant-coefficient, anisotropic stencil in 3D, with symmetry across each axis.
  25-point variable-coefficient, anisotropic stencil in 3D, with symmetry across each axis.

 2nd order in time:
  25-point constant-coefficient, isotropic stencil in 3D, with symmetry across each axis.

  A diagonal dominant matrix is used with "N" as a center-point value for a N-point stencil   
  and "-1/N" as a off-diagonal value.
 */

//conditional selection of stencils to be executed
//#define pt3
#define pt7
#define pt7_var
#define pt25
#define pt25_var
//#define E pt25_t2

using gridtools::level;
using gridtools::accessor;
using gridtools::range;
using gridtools::arg;

namespace cg{

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

// This is the definition of the special regions in the "vertical" direction
// The K dimension can be split into logical splitters. They lie in between array elements.
// A level represent an element on the third coordinate, such that level<2,1> represent
// the array index ofllowing splitter<2>, while lavel<2,-1> represents the array value
// before splitter<2>.
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
typedef gridtools::interval<level<0,-1>, level<1,1> > axis;

// 1st order in time, 3-point constant-coefficient stencil in one dimension, with symmetry.
struct d1point3{
    typedef accessor<0> out;
    typedef const accessor<1, range<-1,1,0,0> > in; // this says to access x-1 anx x+1
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out()) = 3.0*dom(in()) - 1.0/3.0 * (dom(in(x(-1)))+dom(in(x(+1))));
    }
};

// 1st order in time, 7-point constant-coefficient isotropic stencil in 3D, with symmetry.
struct d3point7{
    typedef accessor<0> out;
    typedef const accessor<1, range<-1,1,-1,1> > in; // this says to access 6 neighbors
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out()) = 7.0 * dom(in())
                    - 0.14285714285 * (dom(in(x(-1)))+dom(in(x(+1))))
                    - 0.14285714285 * (dom(in(y(-1)))+dom(in(y(+1))))
                    - 0.14285714285 * (dom(in(z(-1)))+dom(in(z(+1))));
    }
};

// 1st order in time, 7-point variable-coefficient stencil in 3D, with no coefficient symmetry.
struct d3point7_var{
    typedef accessor<0> out;
    typedef const accessor<1, range<-1,1,-1,1> > in; // this says to access 6 neighbors
    typedef const accessor<2, range<0,0,0,0> , 4> coeff;
    typedef boost::mpl::vector<out, in, coeff> arg_list;
    using quad=dimension<4>;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        quad::Index q;
        dom(out()) =  dom(!coeff() * in()
                            + !coeff(q+1) * in(x(-1))
                            + !coeff(q+2) * in(x(+1))
                            + !coeff(q+3) * in(y(-1))
                            + !coeff(q+4) * in(y(+1))
                            + !coeff(q+5) * in(z(-1))
                            + !coeff(q+6) * in(z(+1)));
    }
};

// 25-point, anisotropic stencil in 3D, with symmetry across each axis.
struct d3point25{
    typedef accessor<0> out;
    typedef const accessor<1, range<-4,4,-4,4> > in; // this says to access 24 neighbors
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out()) = 25.0 * dom(in())
                        -0.04 * (dom(in(x(-1))) + dom(in(x(+1))))
                        -0.04 * (dom(in(y(-1))) + dom(in(y(+1))))
                        -0.04 * (dom(in(z(-1))) + dom(in(z(+1))))
                        -0.04 * (dom(in(x(-2))) + dom(in(x(+2))))
                        -0.04 * (dom(in(y(-2))) + dom(in(y(+2))))
                        -0.04 * (dom(in(z(-2))) + dom(in(z(+2))))
                        -0.04 * (dom(in(x(-3))) + dom(in(x(+3))))
                        -0.04 * (dom(in(y(-3))) + dom(in(y(+3))))
                        -0.04 * (dom(in(z(-3))) + dom(in(z(+3))))
                        -0.04 * (dom(in(x(-4))) + dom(in(x(+4))))
                        -0.04 * (dom(in(y(-4))) + dom(in(y(+4))))
                        -0.04 * (dom(in(z(-4))) + dom(in(z(+4))));
    }
};

// 25-point variable-coefficient, anisotropic stencil in 3D, with symmetry across each axis.
struct d3point25_var{
    typedef accessor<0> out;
    typedef const accessor<1, range<-4,4,-4,4> > in; // this says to access 24 neighbors
    typedef const accessor<2, range<0,0,0,0> , 4> coeff;
    typedef boost::mpl::vector<out, in, coeff> arg_list;
    using quad=dimension<4>;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        quad::Index q;
        dom(out()) = dom(!coeff() * in()
                        + !coeff(q+1) * (in(x(-1))+in(x(+1)))
                        + !coeff(q+2) * (in(y(-1))+in(y(+1)))
                        + !coeff(q+3) * (in(z(-1))+in(z(+1)))
                        + !coeff(q+4) * (in(x(-2))+in(x(+2)))
                        + !coeff(q+5) * (in(y(-2))+in(y(+2)))
                        + !coeff(q+6) * (in(z(-2))+in(z(+2)))
                        + !coeff(q+7) * (in(x(-3))+in(x(+3)))
                        + !coeff(q+8) * (in(y(-3))+in(y(+3)))
                        + !coeff(q+9) * (in(z(-3))+in(z(+3)))
                        + !coeff(q+10) * (in(x(-4))+in(x(+4)))
                        + !coeff(q+11) * (in(y(-4))+in(y(+4)))
                        + !coeff(q+12) * (in(z(-4))+in(z(+4))));
    }
};

// 2-nd order in time, 25-point constant-coefficient, isotropic stencil in 3D, with symmetry across each axis.
struct d3point25_time2{
    typedef accessor<0> out;
    typedef const accessor<1, range<-4,4,-4,4> > in; // this says to access 24 neighbors
    typedef const accessor<2> in_old; // this says to access 6 neighbors
    typedef const accessor<3> alpha;
    typedef boost::mpl::vector<out, in, in_old, alpha> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out()) = 2 * dom(in()) - dom(in_old())
                    + dom(alpha()) * (25.0 * dom(in())
                        - 0.04 * (dom(in(x(-1)))+dom(in(x(+1)))
                                    + dom(in(y(-1)))+dom(in(y(+1)))
                                    + dom(in(z(-1)))+dom(in(z(+1))))
                        - 0.04 * (dom(in(x(-2)))+dom(in(x(+2)))
                                    + dom(in(y(-2)))+dom(in(y(+2)))
                                    + dom(in(z(-2)))+dom(in(z(+2))))
                        - 0.04 * (dom(in(x(-3)))+dom(in(x(+3)))
                                    + dom(in(y(-3)))+dom(in(y(+3)))
                                    + dom(in(z(-3)))+dom(in(z(+3))))
                        - 0.04 * (dom(in(x(-4)))+dom(in(x(+4)))
                                    + dom(in(y(-4)))+dom(in(y(+4)))
                                    + dom(in(z(-4)))+dom(in(z(+4))))
                    );
    }
};

/*******************************************************************************/
/*******************************************************************************/

bool solver(uint_t x, uint_t y, uint_t z, uint_t nt) {

    uint_t d1 = x;
    uint_t d2 = y;
    uint_t d3 = z;
    uint_t TIME_STEPS = nt;

    printf("Running for %d x %d x %d, %d iterations\n",x,y,z,nt);

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif

    typedef gridtools::layout_map<0,1,2> layout_t;
    typedef gridtools::layout_map<0,1,2,3> layout4_t;

    typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
    typedef gridtools::BACKEND::storage_type<float_type, layout4_t >::type coeff_type;


    //--------------------------------------------------------------------------
    // Definition of the actual data fields that are used for input/output
#ifdef pt3
    //3pt stencil
    storage_type out3pt(d1,1,1,1., "domain3pt_out");
    storage_type in3pt(d1,1,1,1., "domain3pt_in");
    storage_type *ptr_in3pt = &in3pt, *ptr_out3pt = &out3pt;
#endif

#ifdef pt7
    //7pt stencil with symmetry
    storage_type out7pt(d1,d2,d3,1., "domain7pt_out");
    storage_type in7pt(d1,d2,d3,1., "domain7pt_in");
    storage_type *ptr_in7pt = &in7pt, *ptr_out7pt = &out7pt;
#endif

#ifdef pt7_var
    //7pt stencil with variable coeffs
    storage_type out7pt_var(d1,d2,d3,1., "domain_out");
    storage_type in7pt_var(d1,d2,d3,1., "domain_in");
    storage_type *ptr_in7pt_var = &in7pt_var, *ptr_out7pt_var = &out7pt_var;
    coeff_type coeff7pt_var(d1,d2,d3,7);
    coeff7pt_var.allocate();
    for(uint_t i=0; i<d1; ++i)
        for(uint_t j=0; j<d2; ++j)
            for(uint_t k=0; k<d3; ++k)
                for(uint_t q=0; q<7; ++q)
                {
                    if(q==0) //diagonal point
                        coeff7pt_var(i,j,k,q) = 7.0;
                    else //off-diagonal point
                        coeff7pt_var(i,j,k,q) = -0.14285714285;
                }
#endif

#ifdef pt25
    //25-pt stencil
    storage_type out25pt_const(d1,d2,d3,1., "domain_out");
    storage_type in25pt_const(d1,d2,d3,1., "domain_in");
    storage_type *ptr_in25pt_const = &in25pt_const, *ptr_out25pt_const = &out25pt_const;
#endif   

#ifdef pt25_var
    //25-pt stencil with variable coeffs
    storage_type out25pt_var(d1,d2,d3,1., "domain_out");
    storage_type in25pt_var(d1,d2,d3,1., "domain_in");
    storage_type *ptr_in25pt_var = &in25pt_var, *ptr_out25pt_var = &out25pt_var;
    coeff_type coeff25pt_var(d1,d2,d3,13);
    coeff25pt_var.allocate();
    for(uint_t i=0; i<d1; ++i)
        for(uint_t j=0; j<d2; ++j)
            for(uint_t k=0; k<d3; ++k)
                for(uint_t q=0; q<13; ++q)
                {
                    if(q==0) //diagonal point
                        coeff25pt_var(i,j,k,q) = 25.0;
                    else //off-diagonal point
                        coeff25pt_var(i,j,k,q) = -0.04;
                }
#endif

#ifdef pt25_t2
    //25-pt stencil, 2-nd order in time with coeff symmetry
    storage_type out25pt(d1,d2,d3,1., "domain_out");
    storage_type in25pt(d1,d2,d3,1., "domain_in");
    storage_type in25pt_old(d1,d2,d3,1., "domain_in_old");
    storage_type *ptr_in25pt = &in25pt;
    storage_type *ptr_in25pt_old = &in25pt_old;
    storage_type *ptr_out25pt = &out25pt;
    storage_type alpha(d1,d2,d3,1., "coeff_alpha");
#endif

    //--------------------------------------------------------------------------
    // Definition of placeholders. The order of them reflect the order the user
    // will deal with them especially the non-temporary ones, in the construction
    // of the domain
    typedef arg<0, storage_type > p_out; //domain
    typedef arg<1, storage_type > p_in;
    typedef arg<2, storage_type > p_in_old;
    typedef arg<2, coeff_type > p_coeff;
    typedef arg<3, storage_type > p_alpha;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_out, p_in> accessor_list;
    typedef boost::mpl::vector<p_out, p_in, p_coeff> accessor_list_var;
    typedef boost::mpl::vector<p_out, p_in, p_in_old, p_alpha> accessor_list_time2;

    //--------------------------------------------------------------------------
    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    uint_t dii[5] = {0, 0, 1, d1-2, d1};
    uint_t djj[5] = {0, 0, 0, 0, 1};
    gridtools::coordinates<axis> coords1d3pt(dii, djj);
    coords1d3pt.value_list[0] = 0; //specifying index of the splitter<0,-1>
    coords1d3pt.value_list[1] = 0; //specifying index of the splitter<1,-1>

    //Informs the library that the iteration space in the first two dimensions
    //is from 1 to d1-2 (included) in I (or x) direction
    uint_t di[5] = {0, 0, 1, d1-2, d1};
    //and 1 to d2-2 on J (or y) direction
    uint_t dj[5] = {0, 0, 1, d2-2, d2};
    gridtools::coordinates<axis> coords3d7pt(di, dj);
    coords3d7pt.value_list[0] = 1; //specifying index of the splitter<0,-1>
    coords3d7pt.value_list[1] = d3-2; //specifying index of the splitter<1,-1>

    //domain for 25pt stencil
    uint_t di25[5] = {0, 0, 4, d1-5, d1};
    //and and 0 to 0 on J (or y) direction
    uint_t dj25[5] = {0, 0, 4, d2-5, d2};
    gridtools::coordinates<axis> coords3d25pt(di25, dj25);
    coords3d25pt.value_list[0] = 4; //specifying index of the splitter<0,-1>
    coords3d25pt.value_list[1] = d3-5; //specifying index of the splitter<1,-1>

    /*
      Here we do lot of stuff
      1) We pass to the intermediate representation ::run function the description
      of the stencil, which is a multi-stage stencil (mss)
      The mss includes (in order of execution) a laplacian, two fluxes which are independent
      and a final step that is the out_function
      2) The logical physical domain with the fields to use
      3) The actual domain dimensions
     */

#ifdef pt3
    //start timer
    boost::timer::cpu_times lapse_time1run = {0,0,0};
    boost::timer::cpu_timer time1;

    //TODO: exclude ready, steady,finalize from time measurement (only run)
    for(int i=0; i<TIME_STEPS; i++){

        // construction of the domain
        gridtools::domain_type<accessor_list> domain1d
            (boost::fusion::make_vector(ptr_out3pt, ptr_in3pt));

        //instantiate stencil
        #ifdef __CUDACC__
            gridtools::computation* stencil_step_1 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step_1 =
        #endif
              gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<d1point3>(p_out(), p_in()) // esf_descriptor
                        ),
                    domain1d, coords1d3pt
                    );

        //prepare and run single step of stencil computation
        stencil_step_1->ready();
        stencil_step_1->steady();

        boost::timer::cpu_timer time1run;
        stencil_step_1->run();
        lapse_time1run = lapse_time1run + time1run.elapsed();
        
        stencil_step_1->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out3pt;
        ptr_out3pt = ptr_in3pt;
        ptr_in3pt = tmp;
    }

    boost::timer::cpu_times lapse_time1 = time1.elapsed();
    

#ifdef DEBUG
    printf("Print domain A after computation\n");
    TIME_STEPS % 2 == 0 ? in3pt.print() : out3pt.print();
#endif

    std::cout << "TIME d1point3 TOTAL: " << boost::timer::format(lapse_time1);
    std::cout << "TIME d1point3 RUN:" << boost::timer::format(lapse_time1run) << std::endl;

#endif
//------------------------------------------------------------------------------
#ifdef pt7   
    //start timer
    boost::timer::cpu_times lapse_time2run = {0,0,0};
    boost::timer::cpu_timer time2;

    //TODO: exclude ready, steady,finalize from time measurement (only run)
    for(int i=0; i < TIME_STEPS; i++) {

        // construction of the domain.
        gridtools::domain_type<accessor_list> domain3d
            (boost::fusion::make_vector(ptr_out7pt, ptr_in7pt));

        //instantiate stencil
        #ifdef __CUDACC__
            gridtools::computation* stencil_step_2 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step_2 =
        #endif
              gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<d3point7>(p_out(), p_in()) // esf_descriptor
                        ),
                    domain3d, coords3d7pt
                    );

        //prepare and run single step of stencil computation
        stencil_step_2->ready();
        stencil_step_2->steady();

        boost::timer::cpu_timer time2run;
        stencil_step_2->run();
        lapse_time2run = lapse_time2run + time2run.elapsed();

        stencil_step_2->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out7pt;
        ptr_out7pt = ptr_in7pt;
        ptr_in7pt = tmp;
    }

    boost::timer::cpu_times lapse_time2 = time2.elapsed();


#ifdef DEBUG
    printf("Print domain B after computation\n");
    TIME_STEPS % 2 == 0 ? in7pt.print() : out7pt.print();
#endif

    std::cout << "TIME d3point7 TOTAL: " << boost::timer::format(lapse_time2);
    std::cout << "TIME d3point7 RUN:" << boost::timer::format(lapse_time2run) << std::endl;
#endif
//------------------------------------------------------------------------------
#ifdef pt7_var    
    //start timer
    boost::timer::cpu_times lapse_time3run = {0,0,0};
    boost::timer::cpu_timer time3;

    //TODO: exclude ready, steady,finalize from time measurement (only run)
    for(int i=0; i < TIME_STEPS; i++) {

        // construction of the domain.
        gridtools::domain_type<accessor_list_var> domain3d_var
            (boost::fusion::make_vector(ptr_out7pt_var, ptr_in7pt_var, &coeff7pt_var));

        #ifdef __CUDACC__
            gridtools::computation* stencil_step_3 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step_3 =
        #endif
              gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<d3point7_var>(p_out(), p_in(), p_coeff())
                        ),
                    domain3d_var, coords3d7pt
                    );

        //prepare and run single step of stencil computation
        stencil_step_3->ready();
        stencil_step_3->steady();

        boost::timer::cpu_timer time3run;
        stencil_step_3->run();
        lapse_time3run = lapse_time3run + time3run.elapsed();

        stencil_step_3->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out7pt_var;
        ptr_out7pt_var = ptr_in7pt_var;
        ptr_in7pt_var = tmp;
    }

    boost::timer::cpu_times lapse_time3 = time3.elapsed();

#ifdef DEBUG
    printf("Print domain C after computation\n");
    TIME_STEPS % 2 == 0 ? in7pt_var.print() : out7pt_var.print();
#endif

    std::cout << "TIME d3point7_var TOTAL: " << boost::timer::format(lapse_time3);
    std::cout << "TIME d3point7_var RUN:" << boost::timer::format(lapse_time3run) << std::endl;
#endif
//------------------------------------------------------------------------------
#ifdef pt25
    //start timer
    boost::timer::cpu_times lapse_time40run = {0,0,0};
    boost::timer::cpu_timer time40;

    //TODO: exclude ready, steady,finalize from time measurement (only run)
    for(int i=0; i < TIME_STEPS; i++) {

        // construction of the domain.
        gridtools::domain_type<accessor_list> domain3d_25
            (boost::fusion::make_vector(ptr_out25pt_const, ptr_in25pt_const));

        #ifdef __CUDACC__
            gridtools::computation* stencil_step_40 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step_40 =
        #endif
              gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<d3point25>(p_out(), p_in())
                        ),
                    domain3d_25, coords3d25pt
                    );

        //prepare and run single step of stencil computation
        stencil_step_40->ready();
        stencil_step_40->steady();

        boost::timer::cpu_timer time40run;
        stencil_step_40->run();
        lapse_time40run = lapse_time40run + time40run.elapsed();

        stencil_step_40->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out25pt_const;
        ptr_out25pt_const = ptr_in25pt_const;
        ptr_in25pt_const = tmp;
    }

    boost::timer::cpu_times lapse_time40 = time40.elapsed();

#ifdef DEBUG
    printf("Print domain D0 after computation\n");
    TIME_STEPS % 2 == 0 ? in25pt_const.print() : out25pt_const.print();
#endif

    std::cout << "TIME d3point25 TOTAL: " << boost::timer::format(lapse_time40);
    std::cout << "TIME d3point25 RUN:" << boost::timer::format(lapse_time40run) << std::endl;
#endif
//------------------------------------------------------------------------------
#ifdef pt25_var
    //start timer
    boost::timer::cpu_times lapse_time41run = {0,0,0};
    boost::timer::cpu_timer time41;

    //TODO: exclude ready, steady,finalize from time measurement (only run)
    for(int i=0; i < TIME_STEPS; i++) {

        // construction of the domain.
        gridtools::domain_type<accessor_list_var> domain3d_var25
            (boost::fusion::make_vector(ptr_out25pt_var, ptr_in25pt_var, &coeff25pt_var));

        #ifdef __CUDACC__
            gridtools::computation* stencil_step_41 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step_41 =
        #endif
              gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<d3point25_var>(p_out(), p_in(), p_coeff())
                        ),
                    domain3d_var25, coords3d25pt
                    );

        //prepare and run single step of stencil computation
        stencil_step_41->ready();
        stencil_step_41->steady();

        boost::timer::cpu_timer time41run;
        stencil_step_41->run();
        lapse_time41run = lapse_time41run + time41run.elapsed();

        stencil_step_41->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out25pt_var;
        ptr_out25pt_var = ptr_in25pt_var;
        ptr_in25pt_var = tmp;
    }

    boost::timer::cpu_times lapse_time41 = time41.elapsed();

#ifdef DEBUG
    printf("Print domain D1 after computation\n");
    TIME_STEPS % 2 == 0 ? in25pt_var.print() : out25pt_var.print();
#endif

    std::cout << "TIME d3point25_var TOTAL: " << boost::timer::format(lapse_time41);
    std::cout << "TIME d3point25_var RUN:" << boost::timer::format(lapse_time41run) << std::endl;
#endif
//------------------------------------------------------------------------------
#ifdef pt25_t2
    //start timer
    boost::timer::cpu_times lapse_time5run = {0,0,0};
    boost::timer::cpu_timer time5;

    //TODO: exclude ready, steady,finalize from time measurement (only run)
    for(int i=0; i < TIME_STEPS; i++) {

        // construction of the domain.
        gridtools::domain_type<accessor_list_time2> domain3d_time2
            (boost::fusion::make_vector(ptr_out25pt,ptr_in25pt,
                                        ptr_in25pt_old,&alpha));

        //instantiate stencil
        #ifdef __CUDACC__
            gridtools::computation* stencil_step_5 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step_5 =
        #endif
              gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<d3point25_time2>(p_out(), p_in(),
                                                             p_in_old(), p_alpha())
                        ),
                    domain3d_time2, coords3d25pt
                    );

        //prepare and run single step of stencil computation
        stencil_step_5->ready();
        stencil_step_5->steady();

        boost::timer::cpu_timer time5run;
        stencil_step_5->run();
        lapse_time5run = lapse_time5run + time5run.elapsed();

        stencil_step_5->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out25pt;
        ptr_out25pt = ptr_in25pt_old;
        ptr_in25pt_old = ptr_in25pt;
        ptr_in25pt = tmp;

    }

    boost::timer::cpu_times lapse_time5 = time5.elapsed();

#ifdef DEBUG
    printf("Print domain E after computation\n");
    if(TIME_STEPS % 3 == 0)
        in25pt.print();
    else if(TIME_STEPS % 3 == 1)
        out25pt.print();
    else
        in25pt_old.print();
#endif

    std::cout << "TIME d3point25_time2 TOTAL: " << boost::timer::format(lapse_time5);
    std::cout << "TIME d3point25_time2 RUN:" << boost::timer::format(lapse_time5run) << std::endl;
#endif

    return 1;
    }//solver
}//namespace tridiagonal
