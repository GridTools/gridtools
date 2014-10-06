#if __cplusplus>=201103L

#pragma once


#include <gridtools.h>
#include <stencil-composition/backend_host.h>

#include <boost/timer/timer.hpp>
#include <boost/fusion/include/make_vector.hpp>


using gridtools::level;
using gridtools::arg_type;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;


// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<level<0,-1>, level<1,1> > x_interval;
typedef gridtools::interval<level<0,-2>, level<1,3> > axis;


using namespace expressions;

struct interface{
    static const int n_args = 2;

    typedef arg_type<0> in;
    typedef arg_type<1> out;
    typedef boost::mpl::vector<in, out> arg_list;


    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) {
      dom(in()) = dom(out()+in()/in(z(-1))*out(1,3,1));//dom(out(z(-1))+in()+in()/(in()*in()));//in()+(in()/in()));
    }
};



std::ostream& operator<<(std::ostream& s, interface const) {
    return s << "test_interface";
}


bool test_interface(int x, int y, int z) {

    int d1 = x;
    int d2 = y;
    int d3 = z;

#define BACKEND backend<Host, Naive >

    typedef gridtools::BACKEND::storage_type<double, gridtools::layout_map<0,1,2> >::type storage_type;

     // Definition of the actual data fields that are used for input/output
    storage_type out(d1,d2,d3,2., std::string("out"));
    storage_type in(d1,d2,d3,2., std::string("out"));

    printf("Print OUT field\n");
    out.print();

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef arg<0, storage_type > p_in;
    typedef arg<1, storage_type > p_out;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_in, p_out> arg_type_list;

    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
    gridtools::domain_type<arg_type_list> domain
        (boost::fusion::make_vector(&in, &out));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
    int di[5] = {2, 2, 2, d1-2, d1};
    int dj[5] = {2, 2, 2, d2-2, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3;

    /*
      Here we do lot of stuff
      1) We pass to the intermediate representation ::run function the description
      of the stencil, which is a multi-stage stencil (mss)
      The mss includes (in order of execution) a laplacian, two fluxes which are independent
      and a final step that is the out_function
      2) The logical physical domain with the fields to use
      3) The actual domain dimensions
     */

// \todo simplify the following using the auto keyword from C++11
        boost::shared_ptr<gridtools::computation> forward_step =
        gridtools::make_computation<gridtools::BACKEND>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<interface>(p_in(), p_out()) // esf_descriptor
                ),
            domain, coords
            );


    forward_step->ready();
    forward_step->steady();
    domain.clone_to_gpu();

    boost::timer::cpu_timer time;
    forward_step->run();
    boost::timer::cpu_times lapse_time = time.elapsed();
    forward_step->finalize();

    printf("Print OUT field\n");
    out.print();

    printf("Print IN field\n");
    in.print();

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;
    return        true;
}

#endif //#if __cplusplus>=201103L
