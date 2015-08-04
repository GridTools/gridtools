#include "common/defs.hpp"
#include "stencil-composition/make_stencils.hpp"
#include "stencil-composition/backend.hpp"
#include "stencil-composition/make_computation.hpp"

#include "gtest/gtest.h"

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif


namespace test_staggered_keyword{
    using namespace gridtools;
    using namespace enumtype;

    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;

    struct functor{
        static uint_t ok_i;
        static uint_t ok_j;

        typedef accessor<0> p_i;
        typedef accessor<1> p_j;
        typedef boost::mpl::vector<p_i,p_j> arg_list;
        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval){
            //std::cout<<"i: "<< eval(p_i(-5,-5,0)) <<", j: "<<eval(p_j(-5,-5,0))<< std::endl;
            if(eval(p_i(-5,-5,0))==5)
                ok_i++;
            if(eval(p_j(-5,-5,0))==5)
                ok_j++;
        }
};
uint_t functor::ok_i=0;
uint_t functor::ok_j=0;

bool test(){

    typedef gridtools::layout_map<0,1,2> layout_t;

    typedef gridtools::BACKEND::storage_type<uint_t, layout_t >::type storage_type;

    storage_type i_data ((uint_t) 30,(uint_t) 20, (uint_t) 1);
    storage_type j_data ((uint_t) 30,(uint_t) 20, (uint_t) 1);
    i_data.allocate();
    j_data.allocate();

    i_data.initialize([] (uint_t const& i_, uint_t const& j_, uint_t const& k_) -> uint_t {return i_;});
    j_data.initialize([] (uint_t const& i_, uint_t const& j_, uint_t const& k_) -> uint_t {return j_;});

    uint_t di[5] = {0, 0, 5, 30-1, 30};
    uint_t dj[5] = {0, 0, 5, 20-1, 20};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = 1-1;

    typedef arg<0,storage_type> p_i_data;
    typedef arg<1,storage_type> p_j_data;
    typedef boost::mpl::vector<p_i_data, p_j_data> accessor_list;

    domain_type<accessor_list> domain(boost::fusion::make_vector (&i_data, &j_data));
    auto comp =
        gridtools::make_computation<gridtools::BACKEND, layout_t>
        (
            gridtools::make_mss
            (
                execute<forward>(),
                gridtools::make_esf<functor, staggered<5,5,5,5> >(p_i_data(), p_j_data())
                ),
            domain, coords
            );


    comp->ready();
    comp->steady();
    comp->run();
    return (functor::ok_i&&functor::ok_j);
}
} //namespace test_staggered_keyword

TEST(stencil, test_staggered_keyword){
    EXPECT_TRUE(test_staggered_keyword::test());
}
