#pragma once
#include <stencil-composition/stencil-composition.hpp>

namespace test_expandable_parameters{

    using namespace gridtools;
    using namespace expressions;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

#ifdef CUDA_EXAMPLE
#define BACKEND backend< enumtype::Cuda, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< enumtype::Host, enumtype::Block >
#else
#define BACKEND backend< enumtype::Host, enumtype::Naive >
#endif
#endif

    struct functor_exp{

        typedef vector_accessor<0, enumtype::inout> parameters_out;
        typedef vector_accessor<1, enumtype::in>    parameters_in;
        // typedef accessor<2, enumtype::in> scalar;

        typedef boost::mpl::vector<parameters_out , parameters_in//, scalar
                                   > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const& eval, x_interval){

            eval(parameters_out())=eval(parameters_in());
        }
    };

    struct functor_single_kernel{

        typedef vector_accessor<0, enumtype::inout> parameters1_out;
        typedef vector_accessor<1, enumtype::inout> parameters2_out;
        typedef vector_accessor<2, enumtype::inout> parameters3_out;
        typedef vector_accessor<3, enumtype::inout> parameters4_out;
        typedef vector_accessor<4, enumtype::inout> parameters5_out;
        typedef vector_accessor<5, enumtype::inout> parameters6_out;
        typedef vector_accessor<6, enumtype::inout> parameters7_out;
        typedef vector_accessor<7, enumtype::inout> parameters8_out;

        typedef vector_accessor<8 , enumtype::in>    parameters1_in;
        typedef vector_accessor<9 , enumtype::in>    parameters2_in;
        typedef vector_accessor<10, enumtype::in>    parameters3_in;
        typedef vector_accessor<11, enumtype::in>    parameters4_in;
        typedef vector_accessor<12, enumtype::in>    parameters5_in;
        typedef vector_accessor<13, enumtype::in>    parameters6_in;
        typedef vector_accessor<14, enumtype::in>    parameters7_in;
        typedef vector_accessor<15, enumtype::in>    parameters8_in;
        // typedef accessor<2, enumtype::in> scalar;

        typedef boost::mpl::vector<parameters1_out
                                   , parameters2_out
                                   , parameters3_out
                                   , parameters4_out
                                   , parameters5_out
                                   , parameters6_out
                                   , parameters7_out
                                   , parameters8_out
                                   , parameters1_in
                                   , parameters2_in
                                   , parameters3_in
                                   , parameters4_in
                                   , parameters5_in
                                   , parameters6_in
                                   , parameters7_in
                                   , parameters8_in
                                   > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const& eval, x_interval){

            eval(parameters1_out())=eval(parameters1_in());
            eval(parameters2_out())=eval(parameters2_in());
            eval(parameters3_out())=eval(parameters3_in());
            eval(parameters4_out())=eval(parameters4_in());
            eval(parameters5_out())=eval(parameters5_in());
            eval(parameters6_out())=eval(parameters6_in());
            eval(parameters7_out())=eval(parameters7_in());
            eval(parameters8_out())=eval(parameters8_in());
        }
    };


    struct functor_multi_kernel{

        typedef vector_accessor<0, enumtype::inout> parameters1_out;
        typedef vector_accessor<1, enumtype::inout> parameters2_out;
        typedef vector_accessor<2, enumtype::inout> parameters3_out;
        typedef vector_accessor<3, enumtype::inout> parameters4_out;

        typedef vector_accessor<4, enumtype::in>    parameters1_in;
        typedef vector_accessor<5, enumtype::in>    parameters2_in;
        typedef vector_accessor<6, enumtype::in>    parameters3_in;
        typedef vector_accessor<7, enumtype::in>    parameters4_in;
        // typedef accessor<2, enumtype::in> scalar;

        typedef boost::mpl::vector<parameters1_out
                                   , parameters2_out
                                   , parameters3_out
                                   , parameters4_out
                                   , parameters1_in
                                   , parameters2_in
                                   , parameters3_in
                                   , parameters4_in
                                   > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const& eval, x_interval){

            eval(parameters1_out())=eval(parameters1_in());
            eval(parameters2_out())=eval(parameters2_in());
            eval(parameters3_out())=eval(parameters3_in());
            eval(parameters4_out())=eval(parameters4_in());
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t){

#ifdef CUDA_EXAMPLE
        typedef layout_map<2,1,0> layout_t;
#else
        typedef layout_map<0,1,2> layout_t;
#endif

        typedef BACKEND::storage_info< 23, layout_t > meta_data_t;
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_t;

        typedef storage_t storage_t;

        meta_data_t meta_data_(d1, d2, d3);

        storage_t storage1(meta_data_, 1., "storage1");
        storage_t storage2(meta_data_, 2., "storage2");
        storage_t storage3(meta_data_, 3., "storage3");
        storage_t storage4(meta_data_, 4., "storage4");
        storage_t storage5(meta_data_, 5., "storage5");
        storage_t storage6(meta_data_, 6., "storage6");
        storage_t storage7(meta_data_, 7., "storage7");
        storage_t storage8(meta_data_, 8., "storage8");

        storage_t storage10(meta_data_, -1., "storage10");
        storage_t storage20(meta_data_, -2., "storage20");
        storage_t storage30(meta_data_, -3., "storage30");
        storage_t storage40(meta_data_, -4., "storage40");
        storage_t storage50(meta_data_, -5., "storage50");
        storage_t storage60(meta_data_, -6., "storage60");
        storage_t storage70(meta_data_, -7., "storage70");
        storage_t storage80(meta_data_, -8., "storage80");


        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid_(di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

#ifdef UNROLLED_PARAMETERS

        typedef arg<0, storage_t > p_0_out;
        typedef arg<1, storage_t > p_1_out;
        typedef arg<2, storage_t > p_2_out;
        typedef arg<3, storage_t > p_3_out;
        typedef arg<4, storage_t > p_4_out;
        typedef arg<5, storage_t > p_5_out;
        typedef arg<6, storage_t > p_6_out;
        typedef arg<7, storage_t > p_7_out;

        typedef arg<8 , storage_t > p_0_in;
        typedef arg<9 , storage_t > p_1_in;
        typedef arg<10, storage_t > p_2_in;
        typedef arg<11, storage_t > p_3_in;
        typedef arg<12, storage_t > p_4_in;
        typedef arg<13, storage_t > p_5_in;
        typedef arg<14, storage_t > p_6_in;
        typedef arg<15, storage_t > p_7_in;

        typedef boost::mpl::vector<p_0_out, p_1_out, p_2_out, p_3_out, p_4_out, p_5_out, p_6_out, p_7_out,
                                   p_0_in, p_1_in, p_2_in, p_3_in, p_4_in, p_5_in, p_6_in, p_7_in
                                   > args_t;

        domain_type<args_t> domain_(boost::fusion::make_vector(&storage1, &storage2, &storage3, &storage4, &storage5, &storage6, &storage7, &storage8,
                                                               &storage10, &storage20, &storage30, &storage40, &storage50, &storage60, &storage70, &storage80
                                        ));

#ifdef SINGLE_KERNEL
        auto comp_ = make_computation<BACKEND>(
            domain_, grid_,
                make_mss(
                    enumtype::execute<enumtype::forward>()
                    , make_esf<functor_single_kernel>(p_0_out(), p_1_out(), p_2_out(), p_3_out(), p_4_out(), p_5_out(), p_6_out(), p_7_out(),
                                                      p_0_in(), p_1_in(), p_2_in(), p_3_in(), p_4_in(), p_5_in(), p_6_in(), p_7_in())
                )
            );

        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();

#else // SINGLE_KERNEL

        auto comp1_ = make_computation<BACKEND>(
            domain_, grid_,
                make_mss(
                    enumtype::execute<enumtype::forward>()
                    , make_esf<functor_multi_kernel>(p_0_out(), p_1_out(), p_2_out(), p_3_out(),
                                                     p_0_in(), p_1_in(), p_2_in(), p_3_in())
                )
            );

        auto comp2_ = make_computation<BACKEND>(
            domain_, grid_,
                make_mss(
                    enumtype::execute<enumtype::forward>()
                    , make_esf<functor_multi_kernel>( p_4_out(), p_5_out(), p_6_out(), p_7_out(),
                                                      p_4_in(), p_5_in(), p_6_in(), p_7_in())
                )
            );

        comp1_->ready();
        comp2_->ready();
        comp1_->steady();
        comp2_->steady();
        comp1_->run();
        comp2_->run();
        comp1_->finalize();
        comp2_->finalize();
#endif // SINGLE_KERNEL
#else // UNROLLED_PARAMETERS

        typedef arg<0, std::vector<pointer<storage_t> > > p_list_out;
        typedef arg<1, std::vector<pointer<storage_t> > > p_list_in;

        std::vector<pointer<storage_t> > list_out_={&storage1, &storage2, &storage3, &storage4, &storage5, &storage6, &storage7, &storage8};
        std::vector<pointer<storage_t> > list_in_={&storage10, &storage20, &storage30, &storage40, &storage50, &storage60, &storage70, &storage80};

        typedef boost::mpl::vector<p_list_out, p_list_in> args_t;

        domain_type<args_t> domain_(boost::fusion::make_vector(&list_out_, &list_in_));

        // for(auto &&i:list_out_)
        // {
        //     i->print();
        // }

        auto comp_ = make_computation<BACKEND>(
            expand_factor<4>(), domain_, grid_,
                make_mss(
                    enumtype::execute<enumtype::forward>()
                    , make_esf<functor_exp>(p_list_out(), p_list_in())
                )
            );

        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();

        // for(auto &&i:list_out_)
        // {
        //     i->print();
        // }

#endif
        return true;
    }
}// namespace test_expandable_parameters
