#define PEDANTIC_DISABLED

#include "gtest/gtest.h"
#include "stencil-composition/stencil-composition.hpp"
 
using namespace gridtools;
using namespace enumtype;




typedef interval<level<0,-1>, level<1,-1> > x_interval;
typedef interval<level<0,-2>, level<1,1> > axis;
#ifdef __CUDACC__
typedef backend< Cuda, structured, Block > backend_t;
#ifdef CXX11_ENABLED
typedef backend_t::storage_info< 0, layout_map< 0, 1, 2 > > meta_t;
#else 
typedef meta_storage<
  meta_storage_aligned<
    meta_storage_base<0U, layout_map<0, 1, 2>, false, int, int>, 
    backend_traits_from_id<Cuda>::default_alignment::type, 
    halo<0,0,0> 
  > 
> meta_t;
#endif
#else
typedef backend< Host, structured, Naive > backend_t;
typedef backend_t::storage_info< 0, layout_map< 0, 1, 2 > > meta_t;
#endif
typedef backend_t::storage_type< float_type, meta_t >::type storage_type;

/**@brief generic argument type

   struct implementing the minimal interface in order to be passed as an argument to the user functor.
*/
meta_t global_meta(10,10,10);
struct boundary {
#ifdef _USE_GPU_
    typedef hybrid_pointer< boundary, false > storage_ptr_t;
    typedef hybrid_pointer< const meta_t, false > meta_data_ptr_t;
#else
    typedef wrap_pointer< boundary, false > storage_ptr_t;
    typedef wrap_pointer< const meta_t, false > meta_data_ptr_t;
#endif
    storage_ptr_t m_storage;
    typedef meta_t storage_info_type;
    meta_data_ptr_t m_meta_data;
    boundary() : m_storage(this, true), m_meta_data(&global_meta, true) {}
    ~boundary() {}
    //device copy constructor
    __device__ boundary(const boundary& other){}

    typedef boundary super;
    typedef boundary basic_type;
    typedef boundary* iterator_type;
    typedef boundary value_type; //TODO remove
    static const ushort_t field_dimensions=1; //TODO remove

    GT_FUNCTION
    double value() const {return 10.;}

    template<typename ID>
    GT_FUNCTION
    boundary * access_value() const {return const_cast<boundary*>(this);} //TODO change this?

    GT_FUNCTION
    boundary *get_pointer_to_use() { return m_storage.get_pointer_to_use(); }

    GT_FUNCTION
    pointer< storage_ptr_t > get_storage_pointer() { return pointer< storage_ptr_t >(&m_storage); }

    GT_FUNCTION
    pointer< const storage_ptr_t > get_storage_pointer() const { return pointer< const storage_ptr_t >(&m_storage); }

    GT_FUNCTION
    pointer< const meta_t > get_meta_data_pointer() const { return pointer< const meta_t >(m_meta_data.get_pointer_to_use()); }

    GT_FUNCTION
    void clone_to_device() {
        m_meta_data.update_gpu();
        m_storage.update_gpu();
    }
};

namespace gridtools {
    template <>
    struct is_any_storage< boundary > : boost::mpl::true_ {};
}

struct functor{
    typedef accessor<0, enumtype::inout, extent<0,0,0,0> > sol;
    typedef global_accessor<1, enumtype::in> bd;

    typedef boost::mpl::vector<sol> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        eval(sol())+=eval(bd()).value();
    }
};

TEST(test_global_accessor, boundary_conditions) {
    meta_t meta_(10,10,10);
    storage_type sol_(meta_, (float_type)0.);

    sol_.initialize(2.);

    boundary bd_;

    halo_descriptor di=halo_descriptor(0,1,1,9,10);
    halo_descriptor dj=halo_descriptor(0,1,1,1,2);
    grid<axis> coords_bc(di, dj);
    coords_bc.value_list[0] = 0;
    coords_bc.value_list[1] = 1;

    typedef arg<0, storage_type> p_sol;
    typedef arg<1, boundary> p_bd;

    domain_type<boost::mpl::vector<p_sol, p_bd> > domain ( boost::fusion::make_vector( &sol_, &bd_));

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
