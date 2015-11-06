#define PEDANTIC_DISABLED // too stringent for this test
#include "gtest/gtest.h"
#include <iostream>
#include "common/defs.hpp"
#include "stencil-composition/intermediate_metafunctions.hpp"
#include "stencil-composition/make_computation.hpp"
#include "stencil-composition/backend.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/local_domain.hpp"
#include "stencil-composition/backend_host/iterate_domain_host.hpp"
#include "stencil-composition/accessor.hpp"

namespace test_iterate_domain{
    using namespace gridtools;
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct dummy_functor {
        typedef const accessor<0, range<0,0,0,0>, 6> in;
        typedef const accessor<1, range<0,0,0,0>, 5> buff;
        typedef accessor<2, range<0,0,0,0>, 4> out;
        typedef boost::mpl::vector<in,buff,out> arg_list;


        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };

    std::ostream& operator<<(std::ostream& s, dummy_functor const) {
        return s << "dummy_function";
    }


    bool static test(){
        typedef layout_map<3,2,1,0> layout_ijkp_t;
        typedef layout_map<0,1,2> layout_kji_t;
        typedef layout_map<0,1> layout_ij_t;
    typedef gridtools::backend<enumtype::Host, enumtype::Naive >::storage_type<float_type, layout_ijkp_t >::type storage_type;
    typedef gridtools::backend<enumtype::Host, enumtype::Naive >::storage_type<float_type, layout_kji_t >::type storage_buff_type;
    typedef gridtools::backend<enumtype::Host, enumtype::Naive >::storage_type<float_type, layout_ij_t >::type storage_out_type;

    uint_t d1 = 15;
    uint_t d2 = 13;
    uint_t d3 = 18;
    uint_t d4 = 6;

    field<storage_type, 3, 2, 1>::type in(d1+3,d2+2,d3+1,d4);
    field<storage_buff_type, 4, 7>::type buff(d1,d2,d3);
    field<storage_out_type, 2, 2, 2>::type out(d1+2,d2+1);

    typedef arg<0, field<storage_type, 3, 2, 1>::type > p_in;
    typedef arg<1, field<storage_buff_type, 4, 7>::type > p_buff;
    typedef arg<2, field<storage_out_type, 2, 2, 2>::type > p_out;
    typedef boost::mpl::vector<p_in, p_buff, p_out> accessor_list;


    in.allocate();
    buff.allocate();
    out.allocate();

    gridtools::domain_type<accessor_list> domain((p_in() = in),  (p_buff() = buff), (p_out() = out) );

    uint_t di[5] = {0, 0, 0, d1-1, d1};
    uint_t dj[5] = {0, 0, 0, d2-1, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    auto computation = make_computation<gridtools::backend<enumtype::Host, enumtype::Naive >, layout_ijkp_t>
        (
            gridtools::make_mss // mss_descriptor
            (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<dummy_functor>(p_in() ,p_buff(), p_out())
            ),
            domain, coords
        );

    typedef decltype(gridtools::make_esf<dummy_functor>(p_in() ,p_buff(), p_out())) esf_t;

    computation->ready();
    computation->steady();

    typedef boost::remove_reference<decltype(*computation)>::type intermediate_t;
    typedef intermediate_mss_local_domains<intermediate_t>::type mss_local_domains_t;

    typedef boost::mpl::front<mss_local_domains_t>::type mss_local_domain1_t;

    typedef iterate_domain_host<
        iterate_domain,
        iterate_domain_arguments<
            boost::mpl::at_c<typename mss_local_domain1_t::fused_local_domain_sequence_t, 0>::type,
            boost::mpl::vector1<esf_t>,
            boost::mpl::vector1<range<0,0,0,0> >,
            boost::mpl::vector0<>,
            block_size<32,4>
        >
    > it_domain_t;

    mss_local_domain1_t mss_local_domain1=boost::fusion::at_c<0>(computation->mss_local_domain_list);
    auto local_domain1=boost::fusion::at_c<0>(mss_local_domain1.local_domain_list);
    it_domain_t it_domain(local_domain1);

    GRIDTOOLS_STATIC_ASSERT(it_domain_t::N_STORAGES==3, "bug in iterate domain, incorrect number of storages");

    GRIDTOOLS_STATIC_ASSERT(it_domain_t::N_DATA_POINTERS==23, "bug in iterate domain, incorrect number of data pointers");


    typedef array<void* RESTRICT,it_domain_t::N_DATA_POINTERS> data_pointer_t;
    typedef strides_cached<it_domain_t::N_STORAGES-1, typename decltype(local_domain1)::esf_args> strides_t;
    data_pointer_t data_pointer;
    strides_t strides;

    typedef backend_traits_from_id<enumtype::Host> backend_traits_t;

    it_domain.set_data_pointer_impl(&data_pointer);
    it_domain.set_strides_pointer_impl(&strides);

    it_domain.template assign_storage_pointers<backend_traits_t >();
    it_domain.template assign_stride_pointers <backend_traits_t, strides_t>();

    //check data pointers initialization

    assert(((float_type*)it_domain.data_pointer(0)==in.get<0,0>().get()));
    assert(((float_type*)it_domain.data_pointer(1)==in.get<0,1>().get()));
    assert(((float_type*)it_domain.data_pointer(2)==in.get<0,2>().get()));
    assert(((float_type*)it_domain.data_pointer(3)==in.get<1,0>().get()));
    assert(((float_type*)it_domain.data_pointer(4)==in.get<1,1>().get()));
    assert(((float_type*)it_domain.data_pointer(5)==in.get<2,0>().get()));

    assert(((float_type*)it_domain.data_pointer(6)==buff.get<0,0>().get()));
    assert(((float_type*)it_domain.data_pointer(7)==buff.get<0,1>().get()));
    assert(((float_type*)it_domain.data_pointer(8)==buff.get<0,2>().get()));
    assert(((float_type*)it_domain.data_pointer(9)==buff.get<0,3>().get()));

    assert(((float_type*)it_domain.data_pointer(10)==buff.get<1,0>().get()));
    assert(((float_type*)it_domain.data_pointer(11)==buff.get<1,1>().get()));
    assert(((float_type*)it_domain.data_pointer(12)==buff.get<1,2>().get()));
    assert(((float_type*)it_domain.data_pointer(13)==buff.get<1,3>().get()));
    assert(((float_type*)it_domain.data_pointer(14)==buff.get<1,4>().get()));
    assert(((float_type*)it_domain.data_pointer(15)==buff.get<1,5>().get()));
    assert(((float_type*)it_domain.data_pointer(16)==buff.get<1,6>().get()));

    assert(((float_type*)it_domain.data_pointer(17)==out.get<0,0>().get()));
    assert(((float_type*)it_domain.data_pointer(18)==out.get<0,1>().get()));
    assert(((float_type*)it_domain.data_pointer(19)==out.get<1,0>().get()));
    assert(((float_type*)it_domain.data_pointer(20)==out.get<1,1>().get()));
    assert(((float_type*)it_domain.data_pointer(21)==out.get<2,0>().get()));
    assert(((float_type*)it_domain.data_pointer(22)==out.get<2,1>().get()));

    // check field storage access

    //using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
    it_domain.set_index(0);
    *in.get<0,0>()=0.;//is accessor<0>
    *in.get<0,1>()=1.;
    *in.get<0,2>()=2.;
    *in.get<1,0>()=10.;
    *in.get<1,1>()=11.;
    *in.get<2,0>()=20.;

    assert(it_domain(alias<accessor<0, range<0,0,0,0>, 6>, dimension<5> >::set<0>())==0.);
    assert(it_domain(alias<accessor<0, range<0,0,0,0>, 6>, dimension<5> >::set<1>())==1.);
    assert(it_domain(alias<accessor<0, range<0,0,0,0>, 6>, dimension<5> >::set<2>())==2.);
    assert(it_domain(alias<accessor<0, range<0,0,0,0>, 6>, dimension<6> >::set<1>())==10.);
    assert(it_domain(alias<accessor<0, range<0,0,0,0>, 6>, dimension<6>, dimension<5> >::set<1, 1>())==11.);
    assert(it_domain(alias<accessor<0, range<0,0,0,0>, 6>, dimension<6> >::set<2>())==20.);

    //using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
    *buff.get<0,0>()=0.;//is accessor<1>
    *buff.get<0,1>()=1.;
    *buff.get<0,2>()=2.;
    *buff.get<0,3>()=3.;
    *buff.get<1,0>()=10.;
    *buff.get<1,1>()=11.;
    *buff.get<1,2>()=12.;
    *buff.get<1,3>()=13.;
    *buff.get<1,4>()=14.;
    *buff.get<1,5>()=15.;
    *buff.get<1,6>()=16.;

    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<4> >::set<0>())==0.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<4> >::set<1>())==1.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<4> >::set<2>())==2.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<5> >::set<1>())==10.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 1>())==11.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 2>())==12.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 3>())==13.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 4>())==14.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 5>())==15.);
    assert(it_domain(alias<accessor<1, range<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 6>())==16.);

    *out.get<0,0>()=0.;//is accessor<2>
    *out.get<0,1>()=1.;
    *out.get<1,0>()=10.;
    *out.get<1,1>()=11.;
    *out.get<2,0>()=20.;
    *out.get<2,1>()=21.;

    assert(it_domain(accessor<2, range<0,0,0,0>, 4>())==0.);
    assert(it_domain(accessor<2, range<0,0,0,0>, 4>(dimension<3>(1)))==1.);
    assert(it_domain(accessor<2, range<0,0,0,0>, 4>(dimension<4>(1)))==10.);
    assert(it_domain(accessor<2, range<0,0,0,0>, 4>(dimension<4>(1), dimension<3>(1)))==11.);
    assert(it_domain(accessor<2, range<0,0,0,0>, 4>(dimension<4>(2)))==20.);
    assert(it_domain(accessor<2, range<0,0,0,0>, 4>(dimension<4>(2), dimension<3>(1)))==21.);

    //check index initialization and increment

    array<int_t, 3> index;
    it_domain.get_index(index);
    assert(index[0]==0 && index[1]==0 && index[2]==0);
    index[0] += 3;
    index[1] += 2;
    index[2] += 1;
    it_domain.set_index(index);

    it_domain.get_index(index);
    assert(index[0]==3 && index[1]==2 && index[2]==1);

    array<int_t, 3> new_index;
    it_domain.increment<0,static_uint<1> >();//increment i
    it_domain.increment<1,static_uint<1> >();//increment j
    it_domain.increment<2,static_uint<1> >();//increment k
    it_domain.get_index(new_index);

    //even thought the first case is 4D, we incremented only i,j,k, thus in the check below we don't need the extra stride
    assert(index[0]+in.strides<0>(in.strides())+in.strides<1>(in.strides())+in.strides<2>(in.strides()) == new_index[0] );
    // std::cout<<index[1]<<" + "<<buff.strides<0>(buff.strides()) << " + " << buff.strides<1>(buff.strides()) << " + " << buff.strides<2>(buff.strides())<<std::endl;
    assert(index[1]+buff.strides<0>(buff.strides())+buff.strides<1>(buff.strides())+buff.strides<2>(buff.strides()) == new_index[1] );
    assert(index[2]+out.strides<0>(out.strides())+out.strides<1>(out.strides()) == new_index[2] );

    //check offsets for the space dimensions
    using in_1_1=alias<accessor<0, range<0,0,0,0>, 6>, dimension<6>, dimension<5> >::set<1, 1>;

    assert(((float_type*)(in.get<1,1>().get()+new_index[0]+in.strides<0>(in.strides()))==
            &it_domain(in_1_1(dimension<1>(1)))));

    assert(((float_type*)(in.get<1,1>()+new_index[0]+in.strides<1>(in.strides()))==
            &it_domain(in_1_1(dimension<2>(1)))));

    assert(((float_type*)(in.get<1,1>()+new_index[0]+in.strides<2>(in.strides()))==
            &it_domain(in_1_1(dimension<3>(1)))));

    assert(((float_type*)(in.get<1,1>()+new_index[0]+in.strides<3>(in.strides()))==
            &it_domain(in_1_1(dimension<4>(1)))));

    //check offsets for the space dimensions

    using buff_1_1=alias<accessor<1, range<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 1>;

    assert(((float_type*)(buff.get<1,1>().get()+new_index[1]+buff.strides<0>(buff.strides()))==
            &it_domain(buff_1_1(dimension<1>(1)))));

    assert(((float_type*)(buff.get<1,1>()+new_index[1]+buff.strides<1>(buff.strides()))==
            &it_domain(buff_1_1(dimension<2>(1)))));

    assert(((float_type*)(buff.get<1,1>()+new_index[1]+buff.strides<2>(buff.strides()))==
            &it_domain(buff_1_1(dimension<3>(1)))));


    using out_1=alias<accessor<2, range<0,0,0,0>, 4>, dimension<4>, dimension<3> >::set<1, 1>;

    assert(((float_type*)(out.get<1,1>()+new_index[2]+out.strides<0>(out.strides()))==
            &it_domain(out_1(dimension<1>(1)))));

    assert(((float_type*)(out.get<1,1>()+new_index[2]+out.strides<1>(out.strides()))==
            &it_domain(out_1(dimension<2>(1)))));

    //check strides initialization

    assert(in.strides(1)==strides.get<0>()[0]);
    assert(in.strides(2)==strides.get<0>()[1]);
    assert(in.strides(3)==strides.get<0>()[2]);//4D storage

    assert(buff.strides(1)==strides.get<1>()[0]);
    assert(buff.strides(2)==strides.get<1>()[1]);//3D storage

    assert(out.strides(1)==strides.get<2>()[0]);//2D storage

    return true;
 }
} // namespace test_iterate_domain

TEST(testdomain, iterate_domain) {
    EXPECT_EQ(test_iterate_domain::test(), true);
}
