#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/conditionals/condition_pool.hpp>

namespace test_conditional_switches{
    using namespace gridtools;
    using namespace enumtype;

#define BACKEND_BLOCK

#ifdef CUDA_EXAMPLE
#define BACKEND backend< enumtype::Cuda, GRIDBACKEND, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< enumtype::Host, GRIDBACKEND, enumtype::Block >
#else
#define BACKEND backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >
#endif
#endif

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    template<uint_t Id, typename Extent=extent<> >
    struct functor1{

        typedef accessor<0, enumtype::inout> p_dummy;
        typedef accessor<1, enumtype::inout, Extent> p_dummy_tmp;

        typedef boost::mpl::vector2<p_dummy, p_dummy_tmp> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            eval(p_dummy())+=Id+eval(p_dummy_tmp( 0, 0, 0));
        }
    };

    template<uint_t Id, typename Extent=extent<> >
    struct functor2{

        typedef accessor<0, enumtype::inout> p_dummy;
        typedef accessor<1, enumtype::inout, Extent> p_dummy_tmp;

        typedef boost::mpl::vector2<p_dummy, p_dummy_tmp> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            eval(p_dummy())+=Id+eval(p_dummy_tmp( Extent::iplus::value, 0, 0));
        }
    };

    bool test(){

        bool p = true;
        auto cond_ = new_switch_variable([&p]() { return p ? 0 : 5; });
        auto nested_cond_ = new_switch_variable([]() { return 1; });
        auto other_cond_ = new_switch_variable([&p]() { return p ? 1 : 2; });

        grid<axis> grid_({0,0,0,6,7},{0,0,0,6,7});
        grid_.value_list[0] = 0;
        grid_.value_list[1] = 7;

        typedef gridtools::layout_map<2,1,0> layout_t;//stride 1 on i
        typedef BACKEND::storage_info<__COUNTER__, layout_t> meta_data_t;
        typedef BACKEND::storage_info<__COUNTER__, layout_t> tmp_meta_data_t;
        typedef BACKEND::storage_type<float_type, meta_data_t >::type storage_t;
        typedef BACKEND::temporary_storage_type<float_type, tmp_meta_data_t >::type tmp_storage_t;

        meta_data_t meta_data_(8,8,8);
        storage_t dummy(meta_data_, 0., "dummy");
        typedef arg<0, storage_t > p_dummy;
        typedef arg<1, tmp_storage_t > p_dummy_tmp;

        typedef boost::mpl::vector2<p_dummy, p_dummy_tmp> arg_list;
        domain_type< arg_list > domain_(boost::fusion::make_vector(&dummy));

        auto comp_ = make_computation< BACKEND >(
            domain_,
            grid_,
            make_mss(enumtype::execute< enumtype::forward >(),
                make_esf< functor1< 0 > >(p_dummy(), p_dummy_tmp()),
                make_esf< functor2< 0 > >(p_dummy(), p_dummy_tmp())),
            switch_(cond_,
                case_(0,
                        make_mss(enumtype::execute< enumtype::forward >(),
                            make_esf< functor1< 1, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()),
                            make_esf< functor2< 1, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()))),
                case_(1,
                        make_mss(enumtype::execute< enumtype::forward >(),
                            make_esf< functor1< 2, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()),
                            make_esf< functor2< 2, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()))),
                case_(2,
                        make_mss(enumtype::execute< enumtype::forward >(),
                            make_esf< functor1< 3, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()),
                            make_esf< functor2< 3, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()))),
                case_(3,
                        make_mss(enumtype::execute< enumtype::forward >(),
                            make_esf< functor1< 4, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()),
                            make_esf< functor2< 4, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()))),
                case_(4,
                        make_mss(enumtype::execute< enumtype::forward >(),
                            make_esf< functor1< 5, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()),
                            make_esf< functor2< 5, extent< 0, 1, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()))),
                case_(5,
                        switch_(nested_cond_,
                            case_(2,
                                    make_mss(enumtype::execute< enumtype::forward >(),
                                        make_esf< functor1< 1000 >, extent< 0, 3, 0, 0, 0, 0 > >(
                                                 p_dummy(), p_dummy_tmp()),
                                        make_esf< functor2< 1000 >, extent< 0, 3, 0, 0, 0, 0 > >(
                                                 p_dummy(), p_dummy_tmp()))),
                            case_(1,
                                    make_mss(enumtype::execute< enumtype::forward >(),
                                        make_esf< functor1< 2000, extent< 0, 3, 0, 0, 0, 0 > > >(
                                                 p_dummy(), p_dummy_tmp()),
                                        make_esf< functor2< 2000, extent< 0, 3, 0, 0, 0, 0 > > >(
                                                 p_dummy(), p_dummy_tmp()))),
                            default_(make_mss(enumtype::execute< enumtype::forward >(),
                                make_esf< functor1< 3000, extent< 0, 3, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()),
                                make_esf< functor2< 3000, extent< 0, 3, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()))))),
                case_(6,
                        make_mss(enumtype::execute< enumtype::forward >(),
                            make_esf< functor1< 6, extent< 0, 3, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()),
                            make_esf< functor2< 6, extent< 0, 3, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()))),
                default_(make_mss(enumtype::execute< enumtype::forward >(),
                    make_esf< functor1< 7, extent< 0, 3, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp()),
                    make_esf< functor2< 7, extent< 0, 3, 0, 0, 0, 0 > > >(p_dummy(), p_dummy_tmp())))),
            switch_(other_cond_,
                case_(2,
                        make_mss(enumtype::execute< enumtype::forward >(),
                            make_esf< functor1< 10 > >(p_dummy(), p_dummy_tmp()),
                            make_esf< functor2< 10 > >(p_dummy(), p_dummy_tmp()))),
                case_(1,
                        make_mss(enumtype::execute< enumtype::forward >(),
                            make_esf< functor1< 20 > >(p_dummy(), p_dummy_tmp()),
                            make_esf< functor2< 20 > >(p_dummy(), p_dummy_tmp()))),
                default_(make_mss(enumtype::execute< enumtype::forward >(),
                    make_esf< functor1< 30 > >(p_dummy(), p_dummy_tmp()),
                    make_esf< functor2< 30 > >(p_dummy(), p_dummy_tmp())))),
            make_mss(enumtype::execute< enumtype::forward >(),
                make_esf< functor1< 400 > >(p_dummy(), p_dummy_tmp()),
                make_esf< functor2< 400 > >(p_dummy(), p_dummy_tmp())));

        bool result=true;

        comp_->ready();
        comp_->steady();
        comp_->run();
#ifdef __CUDACC__
        dummy.d2h_update();
#endif
        result = result && (dummy(0,0,0)==842);

        p = false;
        comp_->run();
        comp_->finalize();
        result = result && (dummy(0,0,0)==5662);

        return result;
    }
}//namespace test_conditional

TEST(stencil_composition, conditional_switch){
    EXPECT_TRUE(test_conditional_switches::test());
}
