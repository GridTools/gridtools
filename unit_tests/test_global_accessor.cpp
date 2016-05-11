#define PEDANTIC_DISABLED

#include "gtest/gtest.h"
#include "stencil-composition/stencil-composition.hpp"
using namespace gridtools;
using namespace enumtype;

typedef interval<level<0,-1>, level<1,-1> > x_interval;
typedef interval<level<0,-2>, level<1,1> > axis;


/**@brief generic argument type

   struct implementing the minimal interface in order to be passed as an argument to the user functor.
*/
struct boundary : clonable_to_gpu<boundary> {

    boundary(){}
    //device copy constructor
    __device__ boundary(const boundary& other){}
    typedef boundary super;
    typedef boundary* iterator_type;
    typedef boundary value_type; //TODO remove
    static const ushort_t field_dimensions=1; //TODO remove

    GT_FUNCTION
    double value() const {return 10.;}

    template<typename ID>
    GT_FUNCTION
    boundary * access_value() const {return const_cast<boundary*>(this);} //TODO change this?

    // template<typename ID>
    // boundary const*  access_value() const {return this;} //TODO change this?

};

struct functor{
    typedef accessor<0, enumtype::inout, extent<0,0,0,0> > sol;
    typedef global_accessor<1, enumtype::inout> bd;

    typedef boost::mpl::vector<sol, bd> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        eval(sol())+=eval(bd()).value();
    }
};

TEST(test_global_accessor, boundary_conditions) {

#ifdef __CUDACC__
    typedef backend< Cuda, structured, Block > backend_t;
#else
    typedef backend< Host, structured, Naive > backend_t;
#endif

    typedef typename backend_t::storage_info<0, layout_map<0,1,2> > meta_t;
    meta_t meta_(10,10,10);
    typedef backend_t::storage_type<float_type, meta_t >::type storage_type;
    storage_type sol_(meta_, (float_type)0.);

    sol_.initialize(2.);

    storage_type sol__(meta_, (float_type)0.);

    sol__.initialize(2.);

    boundary bd_;

    halo_descriptor di=halo_descriptor(0,1,1,9,10);
    halo_descriptor dj=halo_descriptor(0,1,1,1,2);
    grid<axis> coords_bc(di, dj);
    coords_bc.value_list[0] = 0;
    coords_bc.value_list[1] = 1;

    typedef arg<0, storage_type> p_sol;
    typedef arg<1, boundary> p_bd;

#ifdef CXX11_ENABLED
    domain_type<boost::mpl::vector<p_sol, p_bd> > domain ((p_sol() = sol_), (p_bd() = bd_));
#else
    domain_type<boost::mpl::vector<p_sol, p_bd> > domain ( boost::fusion::make_vector( &sol_, &bd_));
#endif

#ifdef CXX11_ENABLED
    auto
#else
#ifdef __CUDACC__
    stencil*
#else
        boost::shared_ptr<stencil>
#endif
#endif
        bc_eval = make_computation< backend_t >
        (
            domain, coords_bc
            , make_mss
            (
                execute<forward>(),
                make_esf<functor>(p_sol(), p_bd()))
            );

    bc_eval->ready();
    bc_eval->steady();
    bc_eval->run();
    bc_eval->finalize();

    bool result=true;
    for (int i=0; i<10; ++i)
        for (int j=0; j<10; ++j)
            for (int k=0; k<10; ++k)
            {
                double value=2.;
                if( i>0 && j==1 && k<2)
                    value += 10.;
                if(sol_(i,j,k) != value)
                {
                    result=false;
                }
            }

    EXPECT_TRUE(result);
}
