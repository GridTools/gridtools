#include "gtest/gtest.h"
#include <boost/shared_ptr.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/reductions/reductions.hpp>

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;


namespace red{
#ifdef __CUDACC__
        typedef gridtools::layout_map<2,1,0> layout_t;//stride 1 on i
#else
        //                   strides  1 x xy
        //                      dims  x y z
        typedef gridtools::layout_map<0,1,2> layout_t;//stride 1 on k
#endif

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct sum_red {

        typedef accessor<0, enumtype::in> in;
        typedef boost::mpl::vector<in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static double Do(Evaluation const& eval, x_interval) {
            return eval.reduction_value() + eval(in());
        }
    };

    // These are the stencil operators that compose the multistage stencil in this test
    struct desf {

        typedef accessor<0, enumtype::in> in;
        typedef accessor<1, enumtype::inout> out;
        typedef boost::mpl::vector<in,out> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    void handle_error(int_t)
    {std::cout<<"error"<<std::endl;}

    bool test(uint_t x, uint_t y, uint_t z) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

#ifdef __CUDACC__
#define BACKEND backend<Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, GRIDBACKEND, Block >
#else
#define BACKEND backend<Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef BACKEND::storage_info<__COUNTER__, layout_t> meta_data_t;
        typedef BACKEND::storage_type<float_type, meta_data_t >::type storage_t;

        meta_data_t meta_data_(x,y,z);


        // Definition of the actual data fields that are used for input/output
        typedef storage_t storage_type;
        storage_type in(meta_data_, "in");
        storage_type out(meta_data_, float_type(0.));

        float_type ref=0;
        for(uint_t i=0; i<d1; ++i)
            for(uint_t j=0; j<d2; ++j)
                for(uint_t k=0; k<d3; ++k)
                {
                    in(i,j,k)=i+j+k;
                    ref +=in(i,j,k);
                }

        typedef arg<0, storage_type > p_in;
        typedef arg<1, storage_type > p_out;

        typedef boost::mpl::vector<p_in, p_out> accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
        gridtools::domain_type<accessor_list> domain
            (boost::fusion::make_vector(&in, &out));

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {0, 0, 0, d1-1, d1};
        uint_t dj[5] = {0, 0, 0, d2-1, d2};

        gridtools::grid<axis> grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3-1;

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
        boost::shared_ptr<gridtools::computation> red_ =
            make_computation<gridtools::BACKEND>
            (
                make_mss(
                    execute<forward>(),
                    make_esf<desf>(p_in(),p_out())
                ),
                make_reduction<sum_red, enumtype::plus>(0.0, p_out()),
                domain, grid
            );

        red_->ready();

        red_->steady();

        red_->run();

//#ifdef __CUDACC__
//        out.data().update_cpu();
//#endif

        bool success = true;

        std::cout << "REF " << ref << std::endl;
        return success;
    }
}//namespace red

TEST(reductions, sum) {
    red::test(24, 24, 10);

}
