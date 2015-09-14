#define PEDANTIC_DISABLED

#include "gtest/gtest.h"
#include "stencil-composition/make_computation.hpp"
using namespace gridtools;
using namespace enumtype;

typedef interval<level<0,-1>, level<1,-1> > x_interval;
typedef interval<level<0,-2>, level<1,1> > axis;


/**@brief generic argument type

   struct implementing the minimal interface in order to be passed as an argument to the user functor.
*/
struct boundary{

    boundary(){}
    //device copy constructor
    __device__ boundary(const boundary& other){}
    typedef boundary super;
    typedef boundary* iterator_type;
    typedef boundary value_type; //TODO remove
    static const ushort_t field_dimensions=1; //TODO remove

    double value(){return 10.;}

    template<typename ID>
    boundary * access_value() const {return const_cast<boundary*>(this);} //TODO change this?

};

struct functor{
    typedef accessor<0,range<0,0,0,0> > sol;
    typedef generic_accessor<1> bd;
    typedef boost::mpl::vector<sol> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        eval(sol())+=eval(bd())->value();
    }
};

TEST(test_bc, boundary_conditions) {

    typedef meta_storage<0, layout_map<0,1,2>, false> meta_t;
    meta_t meta_(10,10,10);
    typedef gridtools::backend<Host, Naive>::storage_type<float_type, meta_t >::type storage_type;
    storage_type sol_(meta_);

    sol_.initialize(2.);

    boundary bd_;

    halo_descriptor di=halo_descriptor(0,1,1,9,10);
    halo_descriptor dj=halo_descriptor(0,1,1,1,2);
    coordinates<axis> coords_bc(di, dj);
    coords_bc.value_list[0] = 0;
    coords_bc.value_list[1] = 1;

    typedef arg<0, storage_type> p_sol;
    typedef arg<1, boundary> p_bd;
    domain_type<boost::mpl::vector<p_sol, p_bd> > domain
        (boost::fusion::make_vector(&sol_, &bd_));

    auto bc_eval =
        make_computation<gridtools::backend<Host, Naive> >
        (
            make_mss
            (
                execute<forward>(),
                make_esf<functor>(p_sol(), p_bd()))
            , domain, coords_bc
            );

    for (int i=0; i<10; ++i)
        for (int j=0; j<10; ++j)
            for (int k=0; k<10; ++k)
            {
                double value=2.;
                if(j==0)
                    value += 10.;
                assert(sol_(i,j,k)==value);
            }

}
