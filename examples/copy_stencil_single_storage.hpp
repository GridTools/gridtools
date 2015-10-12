#pragma once

#include <stencil-composition/make_computation.hpp>

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;


namespace copy_stencil{
#ifdef __CUDACC__
        typedef gridtools::layout_map<2,1,0> layout_t;//stride 1 on i
#else
        typedef gridtools::layout_map<0,1,2> layout_t;//stride 1 on k
#endif

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {

        typedef const accessor<0, range<0,0,0,0>, 4> in;
        typedef boost::mpl::vector<in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            eval(in())=eval(in(dimension<4>(1)));
        }

    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream& operator<<(std::ostream& s, copy_functor const) {
        return s << "copy_functor";
    }

    void handle_error(int_t)
    {std::cout<<"error"<<std::endl;}

    typedef storage_info< 0,layout_t> meta_data_t;

    // typedef storage_info_wrapper<meta_storage<0,layout_t, false> > meta_data_t;
    bool test(uint_t x, uint_t y, uint_t z) {

        meta_data_t meta_data_(x,y,z);

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif

        //                   strides  1 x xy
        //                      dims  x y z
        typedef gridtools::BACKEND::storage_type<float_type, meta_data_t >::type storage_t;

        // Definition of the actual data fields that are used for input/output
        typedef field<storage_t,2>::type storage_type;
        storage_type in(meta_data_);
        in.allocate();
        in.initialize(0.);
        for(uint_t i=0; i<d1; ++i){
            for(uint_t j=0; j<d2; ++j){
                for(uint_t k=0; k<d3; ++k){
                    in.template get_value<1,0>(i, j, k)=i+j+k;
                }
            }
        }

        typedef arg<0, storage_type > p_in;

        typedef boost::mpl::vector<
            p_in
            > accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
        gridtools::domain_type<accessor_list> domain
            (boost::fusion::make_vector(&in
                ));

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
        uint_t di[5] = {0, 0, 0, d1-1, d1};
        uint_t dj[5] = {0, 0, 0, d2-1, d2};

        gridtools::coordinates<axis> coords(di, dj);
        coords.value_list[0] = 0;
        coords.value_list[1] = d3-1;

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
#ifdef __CUDACC__
        gridtools::computation* copy =
#else
            boost::shared_ptr<gridtools::computation> copy =
#endif
            gridtools::make_computation<gridtools::BACKEND>
            (
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<copy_functor>(
                        p_in() // esf_descriptor
                        )
                ),
                domain, coords
            );

        copy->ready();

        copy->steady();

        copy->run();

        copy->finalize();

#ifdef BENCHMARK
        std::cout << copy->print_meter() << std::endl;
#endif

        bool success = true;
        for(uint_t i=0; i<d1; ++i){
            for(uint_t j=0; j<d2; ++j){
                for(uint_t k=0; k<d3; ++k){
                    if (in.get_value<0,0>(i, j, k)!=in.get_value<1,0>(i,j,k))
                    {
                        std::cout << "error in "
                                  << i << ", "
                                  << j << ", "
                                  << k << ": "
                                  << "in = " << (in.get_value<0,0>(i, j, k))
                                  << ", out = " << (in.get_value<1,0>(i, j, k))
                                  << std::endl;
                        success = false;
                    }
                }
            }
        }
        if(!success) std::cout << "ERROR" << std::endl;
        else std::cout << "OK" << std::endl;

        return success;
    }
}//namespace copy_stencil
