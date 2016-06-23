#define PEDANTIC_DISABLED // too stringent for this test
#include "gtest/gtest.h"
#include <iostream>
#include "common/defs.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "stencil-composition/intermediate_metafunctions.hpp"

namespace test_iterate_domain{
    using namespace gridtools;
    using namespace enumtype;

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct dummy_functor {
        typedef accessor<0, enumtype::in, extent<0,0,0,0>, 6> in;
        typedef accessor<1, enumtype::in, extent<0,0,0,0>, 5> buff;
        typedef accessor<2, enumtype::inout, extent<0,0,0,0>, 4> out;
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

        typedef gridtools::backend< enumtype::Host, enumtype::structured, enumtype::Naive > backend_t;
        typedef backend_t::storage_info< 0, layout_ijkp_t > meta_ijkp_t;
        typedef backend_t::storage_info< 0, layout_kji_t > meta_kji_t;
        typedef backend_t::storage_info< 0, layout_ij_t > meta_ij_t;

        typedef backend_t::storage_type< float_type, meta_ijkp_t >::type storage_type;
        typedef backend_t::storage_type< float_type, meta_kji_t >::type storage_buff_type;
        typedef backend_t::storage_type< float_type, meta_ij_t >::type storage_out_type;

        uint_t d1 = 15;
        uint_t d2 = 13;
        uint_t d3 = 18;
        uint_t d4 = 6;

        meta_ijkp_t meta_ijkp_(d1+3,d2+2,d3+1,d4);
        field<storage_type, 3, 2, 1>::type in(meta_ijkp_);
        meta_kji_t meta_kji_(d1,d2,d3);
        field<storage_buff_type, 4, 7>::type buff(meta_kji_);
        meta_ij_t meta_ij_(d1+2,d2+1);
        field<storage_out_type, 2, 2, 2>::type out(meta_ij_);

        typedef arg<0, field<storage_type, 3, 2, 1>::type > p_in;
        typedef arg<1, field<storage_buff_type, 4, 7>::type > p_buff;
        typedef arg<2, field<storage_out_type, 2, 2, 2>::type > p_out;
        typedef boost::mpl::vector<p_in, p_buff, p_out> accessor_list;

        gridtools::domain_type<accessor_list> domain((p_in() = in),  (p_buff() = buff), (p_out() = out) );

        uint_t di[5] = {0, 0, 0, d1-1, d1};
        uint_t dj[5] = {0, 0, 0, d2-1, d2};

        gridtools::grid<axis> grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3-1;

        typedef intermediate< gridtools::backend< Host, GRIDBACKEND, Naive >,
            gridtools::meta_array< boost::mpl::vector< decltype(gridtools::make_mss // mss_descriptor
                                       (enumtype::execute< enumtype::forward >(),
                                           gridtools::make_esf< dummy_functor >(p_in(), p_buff(), p_out()))) >,
                                  boost::mpl::quote1< is_amss_descriptor > >,
            decltype(domain),
            decltype(grid),
            boost::fusion::set<>,
            notype,
            false > intermediate_t;

        std::shared_ptr< intermediate_t > computation_ = std::static_pointer_cast< intermediate_t >(
            make_computation< gridtools::backend< Host, GRIDBACKEND, Naive > >(
                domain,
                grid,
                gridtools::make_mss // mss_descriptor
                (enumtype::execute< enumtype::forward >(),
                    gridtools::make_esf< dummy_functor >(p_in(), p_buff(), p_out()))));

        typedef decltype(gridtools::make_esf<dummy_functor>(p_in() ,p_buff(), p_out())) esf_t;

        computation_->ready();
        computation_->steady();

        typedef boost::remove_reference< decltype(*computation_) >::type intermediate_t;
        typedef intermediate_mss_local_domains<intermediate_t>::type mss_local_domains_t;

        typedef boost::mpl::front<mss_local_domains_t>::type mss_local_domain1_t;

        typedef iterate_domain_host<
            iterate_domain,
            iterate_domain_arguments< backend_ids< Host, GRIDBACKEND, Naive >,
                boost::mpl::at_c< typename mss_local_domain1_t::fused_local_domain_sequence_t, 0 >::type,
                boost::mpl::vector1< esf_t >,
                boost::mpl::vector1< extent< 0, 0, 0, 0 > >,
                extent< 0, 0, 0, 0 >,
                boost::mpl::vector0<>,
                block_size< 32, 4 >,
                block_size< 32, 4 >,
                gridtools::grid< axis >,
                boost::mpl::false_,
                notype > > it_domain_t;

        mss_local_domain1_t mss_local_domain1=boost::fusion::at_c<0>(computation_->mss_local_domain_list());
        auto local_domain1=boost::fusion::at_c<0>(mss_local_domain1.local_domain_list);
        it_domain_t it_domain(local_domain1, 0);

        GRIDTOOLS_STATIC_ASSERT(it_domain_t::N_STORAGES==3, "bug in iterate domain, incorrect number of storages");

        GRIDTOOLS_STATIC_ASSERT(it_domain_t::N_DATA_POINTERS==23, "bug in iterate domain, incorrect number of data pointers");

        typename it_domain_t::data_pointer_array_t data_pointer;
        typedef typename it_domain_t::strides_cached_t strides_t;
        strides_t strides;

        typedef backend_traits_from_id< Host > backend_traits_t;

        it_domain.set_data_pointer_impl(&data_pointer);
        it_domain.set_strides_pointer_impl(&strides);

        it_domain.template assign_storage_pointers<backend_traits_t >();
        it_domain.template assign_stride_pointers <backend_traits_t, strides_t>();

        //check data pointers initialization

        assert(((float_type*)it_domain.data_pointer(0)==in.get<0,0>().get()));
        assert(((float_type*)it_domain.data_pointer(1)==in.get<1,0>().get()));
        assert(((float_type*)it_domain.data_pointer(2)==in.get<2,0>().get()));
        assert(((float_type*)it_domain.data_pointer(3)==in.get<0,1>().get()));
        assert(((float_type*)it_domain.data_pointer(4)==in.get<1,1>().get()));
        assert(((float_type*)it_domain.data_pointer(5)==in.get<0,2>().get()));

        assert(((float_type*)it_domain.data_pointer(6)==buff.get<0,0>().get()));
        assert(((float_type*)it_domain.data_pointer(7)==buff.get<1,0>().get()));
        assert(((float_type*)it_domain.data_pointer(8)==buff.get<2,0>().get()));
        assert(((float_type*)it_domain.data_pointer(9)==buff.get<3,0>().get()));

        assert(((float_type*)it_domain.data_pointer(10)==buff.get<0,1>().get()));
        assert(((float_type*)it_domain.data_pointer(11)==buff.get<1,1>().get()));
        assert(((float_type*)it_domain.data_pointer(12)==buff.get<2,1>().get()));
        assert(((float_type*)it_domain.data_pointer(13)==buff.get<3,1>().get()));
        assert(((float_type*)it_domain.data_pointer(14)==buff.get<4,1>().get()));
        assert(((float_type*)it_domain.data_pointer(15)==buff.get<5,1>().get()));
        assert(((float_type*)it_domain.data_pointer(16)==buff.get<6,1>().get()));

        assert(((float_type*)it_domain.data_pointer(17)==out.get<0,0>().get()));
        assert(((float_type*)it_domain.data_pointer(18)==out.get<1,0>().get()));
        assert(((float_type*)it_domain.data_pointer(19)==out.get<0,1>().get()));
        assert(((float_type*)it_domain.data_pointer(20)==out.get<1,1>().get()));
        assert(((float_type*)it_domain.data_pointer(21)==out.get<0,2>().get()));
        assert(((float_type*)it_domain.data_pointer(22)==out.get<1,2>().get()));

        // check field storage access

        //using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
        it_domain.set_index(0);
        *in.get<0,0>()=0.;//is accessor<0>
        *in.get<1,0>()=1.;
        *in.get<2,0>()=2.;
        *in.get<0,1>()=10.;
        *in.get<1,1>()=11.;
        *in.get<0,2>()=20.;

        assert(it_domain(alias<inout_accessor<0, extent<0,0,0,0>, 6>, dimension<5> >::set<0>())==0.);
        assert(it_domain(alias<inout_accessor<0, extent<0,0,0,0>, 6>, dimension<5> >::set<1>())==1.);
        assert(it_domain(alias<inout_accessor<0, extent<0,0,0,0>, 6>, dimension<5> >::set<2>())==2.);
        assert(it_domain(alias<inout_accessor<0, extent<0,0,0,0>, 6>, dimension<6> >::set<1>())==10.);
        assert(it_domain(alias<inout_accessor<0, extent<0,0,0,0>, 6>, dimension<6>, dimension<5> >::set<1, 1>())==11.);
        assert(it_domain(alias<inout_accessor<0, extent<0,0,0,0>, 6>, dimension<6> >::set<2>())==20.);

        //using compile-time constexpr accessors (through alias::set) when the data field is not "rectangular"
        *buff.get<0,0>()=0.;//is accessor<1>
        *buff.get<1,0>()=1.;
        *buff.get<2,0>()=2.;
        *buff.get<3,0>()=3.;
        *buff.get<0,1>()=10.;
        *buff.get<1,1>()=11.;
        *buff.get<2,1>()=12.;
        *buff.get<3,1>()=13.;
        *buff.get<4,1>()=14.;
        *buff.get<5,1>()=15.;
        *buff.get<6,1>()=16.;

        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<4> >::set<0>())==0.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<4> >::set<1>())==1.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<4> >::set<2>())==2.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<5> >::set<1>())==10.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 1>())==11.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 2>())==12.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 3>())==13.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 4>())==14.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 5>())==15.);
        assert(it_domain(alias<accessor<1, enumtype::in,extent<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 6>())==16.);

        *out.get<0,0>()=0.;//is accessor<2>
        *out.get<1,0>()=1.;
        *out.get<0,1>()=10.;
        *out.get<1,1>()=11.;
        *out.get<0,2>()=20.;
        *out.get<1,2>()=21.;



        assert(it_domain(accessor<2, enumtype::inout,extent<0,0,0,0>, 4>())==0.);
        assert(it_domain(accessor<2, enumtype::inout,extent<0,0,0,0>, 4>(dimension<3>(1)))==1.);
        assert(it_domain(accessor<2, enumtype::inout,extent<0,0,0,0>, 4>(dimension<4>(1)))==10.);
        assert(it_domain(accessor<2, enumtype::inout,extent<0,0,0,0>, 4>(dimension<4>(1), dimension<3>(1)))==11.);
        assert(it_domain(accessor<2, enumtype::inout,extent<0,0,0,0>, 4>(dimension<4>(2)))==20.);
        assert(it_domain(accessor<2, enumtype::inout,extent<0,0,0,0>, 4>(dimension<4>(2), dimension<3>(1)))==21.);


        assert(it_domain(inout_accessor<2, extent<0,0,0,0>, 4>(0,0,0,0) ) == 0.);
        assert(it_domain(inout_accessor<2, extent<0,0,0,0>, 4>(0,0,1,0) ) == 1.);
        assert(it_domain(inout_accessor<2, extent<0,0,0,0>, 4>(0,0,0,1) ) == 10.);
        assert(it_domain(inout_accessor<2, extent<0,0,0,0>, 4>(0,0,1,1) ) == 11.);
        assert(it_domain(inout_accessor<2, extent<0,0,0,0>, 4>(0,0,0,2) ) == 20.);
        assert(it_domain(inout_accessor<2, extent<0,0,0,0>, 4>(0,0,1,2) ) == 21.);

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
        assert(index[2]+in.meta_data().strides<0>(in.meta_data().strides())+in.meta_data().strides<1>(in.meta_data().strides())+in.meta_data().strides<2>(in.meta_data().strides()) == new_index[2] );
        // std::cout<<index[1]<<" + "<<buff.strides<0>(buff.strides()) << " + " << buff.strides<1>(buff.strides()) << " + " << buff.strides<2>(buff.strides())<<std::endl;
        assert(index[1]+buff.meta_data().strides<0>(buff.meta_data().strides())+buff.meta_data().strides<1>(buff.meta_data().strides())+buff.meta_data().strides<2>(buff.meta_data().strides()) == new_index[1] );
        assert(index[0]+out.meta_data().strides<0>(out.meta_data().strides())+out.meta_data().strides<1>(out.meta_data().strides()) == new_index[0] );

        //check offsets for the space dimensions
        using in_1_1=alias<accessor<0, enumtype::inout,extent<0,0,0,0>, 6>, dimension<6>, dimension<5> >::set<1, 1>;

        assert(((float_type*)(in.get<1,1>().get()+new_index[2]+in.meta_data().strides<0>(in.meta_data().strides()))==
                &it_domain(in_1_1(dimension<1>(1)))));

        assert(((float_type*)(in.get<1,1>()+new_index[2]+in.meta_data().strides<1>(in.meta_data().strides()))==
                &it_domain(in_1_1(dimension<2>(1)))));

        assert(((float_type*)(in.get<1,1>()+new_index[2]+in.meta_data().strides<2>(in.meta_data().strides()))==
                &it_domain(in_1_1(dimension<3>(1)))));

        assert(((float_type*)(in.get<1,1>()+new_index[2]+in.meta_data().strides<3>(in.meta_data().strides()))==
                &it_domain(in_1_1(dimension<4>(1)))));

        //check offsets for the space dimensions

        using buff_1_1=alias<accessor<1, enumtype::inout,extent<0,0,0,0>, 5>, dimension<5>, dimension<4> >::set<1, 1>;

        assert(((float_type*)(buff.get<1,1>().get()+new_index[1]+buff.meta_data().strides<0>(buff.meta_data().strides()))==
                &it_domain(buff_1_1(dimension<1>(1)))));

        assert(((float_type*)(buff.get<1,1>()+new_index[1]+buff.meta_data().strides<1>(buff.meta_data().strides()))==
                &it_domain(buff_1_1(dimension<2>(1)))));

        assert(((float_type*)(buff.get<1,1>()+new_index[1]+buff.meta_data().strides<2>(buff.meta_data().strides()))==
                &it_domain(buff_1_1(dimension<3>(1)))));

        using out_1=alias<inout_accessor<2, extent<0,0,0,0>, 4>, dimension<4>, dimension<3> >::set<1, 1>;

        assert(((float_type*)(out.get<1,1>()+new_index[0]+out.meta_data().strides<0>(out.meta_data().strides()))==
                &it_domain(out_1(dimension<1>(1)))));

        assert(((float_type*)(out.get<1,1>()+new_index[0]+out.meta_data().strides<1>(out.meta_data().strides()))==
                &it_domain(out_1(dimension<2>(1)))));

        //check strides initialization

        assert(in.meta_data().strides(1)==strides.get<2>()[0]);
        assert(in.meta_data().strides(2)==strides.get<2>()[1]);
        assert(in.meta_data().strides(3)==strides.get<2>()[2]);//4D storage

        assert(buff.meta_data().strides(1)==strides.get<1>()[0]);
        assert(buff.meta_data().strides(2)==strides.get<1>()[1]);//3D storage

        assert(out.meta_data().strides(1)==strides.get<0>()[0]);//2D storage

        return true;
    }
} // namespace test_iterate_domain

TEST(testdomain, iterate_domain) {
    EXPECT_EQ(test_iterate_domain::test(), true);
}
