#pragma once

#include <gridtools.hpp>

#include <stencil-composition/backend.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>

#include <boost/timer/timer.hpp>

/*
  @file This file shows an implementation of the various stencil operations.

 1nd order in time:
  3-point constant-coefficient stencil in one dimension, with symmetry.
  7-point constant-coefficient isotropic stencil in three dimensions, with symmetry.
  7-point variable-coefficient stencil in three dimension, with no coefficient symmetry.
  25-point variable-coefficient, anisotropic stencil in 3D, with symmetry across each axis.

 2nd order in time:
  25-point constant-coefficient, isotropic stencil in 3D, with symmetry across each axis.

  A diagonal dominant matrix is used with "N" as a center-point value for a N-point stencil   
  and "-1/N" as a off-diagonal value.
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::range;
using gridtools::arg;

namespace cg{

using namespace gridtools;
using namespace enumtype;

#ifdef CXX11_ENABLED
using namespace expressions;
#endif

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
    typedef accessor<1, range<-1,1,0,0> > in; // this says to access x-1 anx x+1
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
    typedef accessor<1, range<-1,1,-1,1> > in; // this says to access 6 neighbors
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out()) = 7.0 * dom(in())
                    - 1.0/7.0 * (dom(in(x(-1)))+dom(in(x(+1))))
                    - 1.0/7.0 * (dom(in(y(-1)))+dom(in(y(+1))))
                    - 1.0/7.0 * (dom(in(z(-1)))+dom(in(z(+1))));
    }
};

// 1st order in time, 7-point variable-coefficient stencil in 3D, with no coefficient symmetry.
struct d3point7_var{
    typedef accessor<0> out;
    typedef accessor<1, range<-1,1,-1,1> > in; // this says to access 6 neighbors
    typedef accessor<2, range<0,0,0,0> , 4> const coeff;
    typedef boost::mpl::vector<out, in, coeff> arg_list;
    using quad=Dimension<4>;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        x::Index i;
        y::Index j;
        z::Index k;
        quad::Index q;
        dom(out()) =  dom(!coeff(i,j,k,q) * in()
                            + !coeff(i,j,k,q+1) * in(x(-1))
                            + !coeff(i,j,k,q+2) * in(x(+1))
                            + !coeff(i,j,k,q+3) * in(y(-1))
                            + !coeff(i,j,k,q+4) * in(y(+1))
                            + !coeff(i,j,k,q+5) * in(z(-1))
                            + !coeff(i,j,k,q+6) * in(z(+1)));
    }
};

// 25-point variable-coefficient, anisotropic stencil in 3D, with symmetry across each axis.
struct d3point25_var{
    typedef accessor<0> out;
    typedef accessor<1, range<-4,4,-4,4> > in; // this says to access 24 neighbors
    typedef accessor<2> a;
    typedef accessor<3> b;
    typedef accessor<4> c;
    typedef accessor<5> d;
    typedef accessor<6> e;
    typedef accessor<7> f;
    typedef accessor<8> g;
    typedef accessor<9> h;
/*    typedef accessor<10> i;
    typedef accessor<11> j;
    typedef accessor<12> k;
    typedef accessor<13> l;
    typedef accessor<14> m; */
    typedef boost::mpl::vector<out, in, a, b, c, d, e, f, g, h/*, i, j, k, l, m*/> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out()) = dom(a()) * dom(in())
                    + dom(b()) * (dom(in(x(-1)))+dom(in(x(+1))))
                    + dom(c()) * (dom(in(y(-1)))+dom(in(y(+1))))
                    + dom(d()) * (dom(in(z(-1)))+dom(in(z(+1))))
                    + dom(e()) * (dom(in(x(-2)))+dom(in(x(+2))))
                    + dom(f()) * (dom(in(y(-2)))+dom(in(y(+2))))
                    + dom(g()) * (dom(in(z(-2)))+dom(in(z(+2))))
                    + dom(h()) * (dom(in(x(-3)))+dom(in(x(+3))))
                    + dom(h()) * (dom(in(y(-3)))+dom(in(y(+3))))
                    + dom(h()) * (dom(in(z(-3)))+dom(in(z(+3))))
                    + dom(h()) * (dom(in(x(-4)))+dom(in(x(+4))))
                    + dom(h()) * (dom(in(y(-4)))+dom(in(y(+4))))
                    + dom(h()) * (dom(in(z(-4)))+dom(in(z(+4))));
    }
};

// 2-nd order in time, 25-point constant-coefficient, isotropic stencil in 3D, with symmetry across each axis.
struct d3point25_time2{
    typedef accessor<0> out;
    typedef accessor<1, range<-4,4,-4,4> > in; // this says to access 24 neighbors
    typedef accessor<2> in_old; // this says to access 6 neighbors
    typedef accessor<3> alpha;
    typedef boost::mpl::vector<out, in, in_old, alpha> arg_list;

    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
        dom(out()) = 2 * dom(in()) - dom(in_old())
                    + dom(alpha()) * (25.0 * dom(in())
                        - 1/25.0 * (dom(in(x(-1)))+dom(in(x(+1)))
                                    + dom(in(y(-1)))+dom(in(y(+1)))
                                    + dom(in(z(-1)))+dom(in(z(+1))))
                        - 1/25.0 * (dom(in(x(-2)))+dom(in(x(+2)))
                                    + dom(in(y(-2)))+dom(in(y(+2)))
                                    + dom(in(z(-2)))+dom(in(z(+2))))
                        - 1/25.0 * (dom(in(x(-3)))+dom(in(x(+3)))
                                    + dom(in(y(-3)))+dom(in(y(+3)))
                                    + dom(in(z(-3)))+dom(in(z(+3))))
                        - 1/25.0 * (dom(in(x(-4)))+dom(in(x(+4)))
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
    //3pt stencil
    storage_type out3pt(d1,1,1,1., "domain3pt_out");
    storage_type in3pt(d1,1,1,1., "domain3pt_in");
    storage_type *ptr_in3pt = &in3pt, *ptr_out3pt = &out3pt;

    //7pt stencil with symmetry
    storage_type out7pt(d1,d2,d3,1., "domain7pt_out");
    storage_type in7pt(d1,d2,d3,1., "domain7pt_in");
    storage_type *ptr_in7pt = &in7pt, *ptr_out7pt = &out7pt;

    //7pt stencil with variable coeffs
    storage_type out7pt_var(d1,d2,d3,1., "domain_out");
    storage_type in7pt_var(d1,d2,d3,1., "domain_in");
    storage_type *ptr_in7pt_var = &in7pt_var, *ptr_out7pt_var = &out7pt_var;
    coeff_type coeff7pt_var(d1,d2,d3,7);
    coeff7pt_var.allocate();
    /*for(uint_t i=0; i<d1; ++i)
        for(uint_t j=0; j<d2; ++j)
            for(uint_t k=0; k<d3; ++k)
                for(uint_t q=0; q<7; ++q)
                {
                    if(q==0) //diagonal point
                        coeff7pt_var(i,j,k,q) = 7.0;
                    else //off-diagonal point
                        coeff7pt_var(i,j,k,q) = -1/7.0;
                }*/
   

    //25-pt stencil with variable coeffs
    storage_type out25pt_var(d1,d2,d3,1., "domain_out");
    storage_type in25pt_var(d1,d2,d3,1., "domain_in");
    storage_type *ptr_in25pt_var = &in25pt_var, *ptr_out25pt_var = &out25pt_var;
    storage_type a_var25(d1,d2,d3,25., "coeff_a");
    storage_type b_var25(d1,d2,d3,-1/25., "coeff_b");
    storage_type c_var25(d1,d2,d3,-1/25., "coeff_c");
    storage_type d_var25(d1,d2,d3,-1/25., "coeff_d");
    storage_type e_var25(d1,d2,d3,-1/25., "coeff_e");
    storage_type f_var25(d1,d2,d3,-1/25., "coeff_f");
    storage_type g_var25(d1,d2,d3,-1/25., "coeff_g");
    storage_type h_var25(d1,d2,d3,-1/25., "coeff_h");
    //TODO encapsulate coeffs - boost::make_vector takes max 10 parameters
    /*storage_type i_var25(d1,d2,d3,-1/25., "coeff_i");
    storage_type j_var25(d1,d2,d3,-1/25., "coeff_j");
    storage_type k_var25(d1,d2,d3,-1/25., "coeff_k");
    storage_type l_var25(d1,d2,d3,-1/25., "coeff_l");
    storage_type m_var25(d1,d2,d3,-1/25., "coeff_m");*/

    //25-pt stencil, 2-nd order in time with coeff symmetry
    storage_type out25pt(d1,d2,d3,1., "domain_out");
    storage_type in25pt(d1,d2,d3,1., "domain_in");
    storage_type in25pt_old(d1,d2,d3,1., "domain_in_old");
    storage_type *ptr_in25pt = &in25pt;
    storage_type *ptr_in25pt_old = &in25pt_old;
    storage_type *ptr_out25pt = &out25pt;
    storage_type alpha(d1,d2,d3,1., "coeff_alpha");

    //--------------------------------------------------------------------------
    // Definition of placeholders. The order of them reflect the order the user
    // will deal with them especially the non-temporary ones, in the construction
    // of the domain
    typedef arg<0, storage_type > p_out; //domain
    typedef arg<1, storage_type > p_in;
    typedef arg<2, storage_type > p_in_old;
    typedef arg<2, coeff_type > p_coeff;
    typedef arg<2, storage_type > p_a;
    typedef arg<3, storage_type > p_alpha;
    typedef arg<3, storage_type > p_b;
    typedef arg<4, storage_type > p_c;
    typedef arg<5, storage_type > p_d;
    typedef arg<6, storage_type > p_e;
    typedef arg<7, storage_type > p_f;
    typedef arg<8, storage_type > p_g;
    typedef arg<9, storage_type > p_h;
/*    typedef arg<10, storage_type > p_i;
    typedef arg<11, storage_type > p_j;
    typedef arg<12, storage_type > p_k;
    typedef arg<13, storage_type > p_l;
    typedef arg<14, storage_type > p_m;*/

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_out, p_in> accessor_list;
    typedef boost::mpl::vector<p_out, p_in, p_coeff> accessor_list_var7pt;
    typedef boost::mpl::vector<p_out, p_in, p_a, p_b, p_c, p_d, p_e, p_f, p_g,
                               p_h/*, p_i, p_j, p_k, p_l, p_m*/> accessor_list_var25pt;
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

    //start timer
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
        stencil_step_1->run();
        stencil_step_1->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out3pt;
        ptr_out3pt = ptr_in3pt;
        ptr_in3pt = tmp;
    }

    boost::timer::cpu_times lapse_time1 = time1.elapsed();
    

    printf("Print domain A after computation\n");
    TIME_STEPS % 2 == 0 ? in3pt.print() : out3pt.print();

    std::cout << "TIME d1point3: " << boost::timer::format(lapse_time1) << std::endl;
//------------------------------------------------------------------------------
   
    //start timer
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
        stencil_step_2->run();
        stencil_step_2->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out7pt;
        ptr_out7pt = ptr_in7pt;
        ptr_in7pt = tmp;
    }

    boost::timer::cpu_times lapse_time2 = time2.elapsed();


    printf("Print domain B after computation\n");
    TIME_STEPS % 2 == 0 ? in7pt.print() : out7pt.print();

    std::cout << "TIME d3point7: " << boost::timer::format(lapse_time2) << std::endl;
//------------------------------------------------------------------------------
    
    //start timer
    boost::timer::cpu_timer time3;

    //TODO: exclude ready, steady,finalize from time measurement (only run)
    for(int i=0; i < TIME_STEPS; i++) {

        // construction of the domain.
        gridtools::domain_type<accessor_list_var7pt> domain3d_var
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
        stencil_step_3->run();
        stencil_step_3->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out7pt_var;
        ptr_out7pt_var = ptr_in7pt_var;
        ptr_in7pt_var = tmp;
    }

    boost::timer::cpu_times lapse_time3 = time3.elapsed();

    printf("Print domain C after computation\n");
    TIME_STEPS % 2 == 0 ? in7pt_var.print() : out7pt_var.print();

    std::cout << "TIME d3point7_var: " << boost::timer::format(lapse_time3) << std::endl;
//------------------------------------------------------------------------------

    //start timer
    boost::timer::cpu_timer time4;

    //TODO: exclude ready, steady,finalize from time measurement (only run)
    for(int i=0; i < TIME_STEPS; i++) {

        // construction of the domain.
        gridtools::domain_type<accessor_list_var25pt> domain3d_var25
            (boost::fusion::make_vector(ptr_out25pt_var, ptr_in25pt_var, &a_var25, &b_var25, &c_var25, &d_var25, &e_var25, &f_var25, &g_var25, &h_var25/*, &i_var25, &j_var25, &k_var25, &l_var25, &m_var25*/));

        #ifdef __CUDACC__
            gridtools::computation* stencil_step_4 =
        #else
                boost::shared_ptr<gridtools::computation> stencil_step_4 =
        #endif
              gridtools::make_computation<gridtools::BACKEND, layout_t>
                (
                    gridtools::make_mss // mss_descriptor
                    (
                        execute<forward>(),
                        gridtools::make_esf<d3point25_var>(p_out(), p_in(), p_a(),
                                                          p_b(), p_c(), p_d(),
                                                          p_e(), p_f(), p_g(),
                                                          p_h()/*, p_i(), p_j(),
                                                          p_k(), p_l(), p_m()*/) // esf_descriptor
                        ),
                    domain3d_var25, coords3d25pt
                    );

        //prepare and run single step of stencil computation
        stencil_step_4->ready();
        stencil_step_4->steady();
        stencil_step_4->run();
        stencil_step_4->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out25pt_var;
        ptr_out25pt_var = ptr_in25pt_var;
        ptr_in25pt_var = tmp;
    }

    boost::timer::cpu_times lapse_time4 = time4.elapsed();

    printf("Print domain D after computation\n");
    TIME_STEPS % 2 == 0 ? in25pt_var.print() : out25pt_var.print();

    std::cout << "TIME d3point25_var: " << boost::timer::format(lapse_time4) << std::endl;
//------------------------------------------------------------------------------

    //start timer
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
        stencil_step_5->run();
        stencil_step_5->finalize();

        //swap input and output fields
        storage_type* tmp = ptr_out25pt;
        ptr_out25pt = ptr_in25pt_old;
        ptr_in25pt_old = ptr_in25pt;
        ptr_in25pt = tmp;

    }

    boost::timer::cpu_times lapse_time5 = time5.elapsed();


    printf("Print domain E after computation\n");
    if(TIME_STEPS % 3 == 0)
        in25pt.print();
    else if(TIME_STEPS % 3 == 1)
        out25pt.print();
    else
        in25pt_old.print();

    std::cout << "TIME d3point25_time2: " << boost::timer::format(lapse_time5) << std::endl;


    return 1;
    }//solver
}//namespace tridiagonal
