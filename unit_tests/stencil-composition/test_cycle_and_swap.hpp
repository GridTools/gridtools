#pragma once
//disabling pedantic mode because I want to use a 2D layout map
//(to test the case in which the 3rd dimension is not k)
#define PEDANTIC_DISABLED

#include <stencil-composition/make_computation.hpp>

namespace test_cycle_and_swap{
    using namespace gridtools;
    using namespace enumtype;

    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;

    struct functor{
        typedef accessor<0, range<>, 3> p_i;
        typedef boost::mpl::vector<p_i> arg_list;
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval){
            eval(p_i())+=eval(p_i());
        }
    };


#ifdef __CUDACC__
#define BACKEND backend<Cuda, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif

    bool test(){

        typedef gridtools::layout_map<0,1> layout_t;
        typedef gridtools::storage_info<0, layout_t> meta_t;
        typedef gridtools::BACKEND::storage_type<uint_t, meta_t >::type storage_type;
        typedef typename field<storage_type, 2>::type field_t;

        meta_t meta_( 1, 1);
        field_t i_data (meta_);
        i_data.allocate();
        i_data.get_value<0,0>(0,0)=0.;
        i_data.get_value<1,0>(0,0)=1.;

        uint_t di[5] = {0, 0, 0, 0, 1};
        uint_t dj[5] = {0, 0, 0, 0, 1};

        gridtools::coordinates<axis> coords(di, dj);
        coords.value_list[0] = 0;
        coords.value_list[1] = 0;

        typedef arg<0,field_t> p_i_data;
        typedef boost::mpl::vector<p_i_data> accessor_list;

        domain_type<accessor_list> domain(boost::fusion::make_vector (&i_data));

        auto comp =
            gridtools::make_computation<gridtools::BACKEND>
            (
                gridtools::make_mss
                (
                    execute<forward>(),
                    gridtools::make_esf<functor>(p_i_data())
                    ),
                domain, coords
                );


        comp->ready();
        comp->steady();
        comp->run();
        swap<0,0>::with<1,0>::apply(i_data);
        comp->run();
        comp->finalize();

        return (i_data(0,0)==2 && i_data.get_value<1,0>(0,0)==0);
    }
} //namespace test_cycle_and_swap
