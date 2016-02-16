#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

namespace test_conditional_switches{
    using namespace gridtools;


#ifdef CUDA_EXAMPLE
#define BACKEND backend<enumtype::Cuda, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<enumtype::Host, enumtype::Block >
#else
#define BACKEND backend<enumtype::Host, enumtype::Naive >
#endif
#endif

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    template<uint_t Id>
    struct functor{

        typedef accessor<0> p_dummy;
        typedef boost::mpl::vector1<p_dummy> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            printf("%d\n", Id);
        }
    };

    int test(){

        switch_variable<0,int> cond(0);
        switch_variable<0,int> new_cond(5);
        switch_variable<3,int> nested_cond_(1);
        switch_variable<1,int> other_cond_(1);
        switch_variable<1,int> new_other_cond_(2);

        grid<axis> grid_({0,0,0,1,2},{0,0,0,1,2});
        grid_.value_list[0] = 0;
        grid_.value_list[1] = 2;

        typedef gridtools::layout_map<2,1,0> layout_t;//stride 1 on i
        typedef BACKEND::storage_info<__COUNTER__, layout_t> meta_data_t;
        typedef BACKEND::storage_type<float_type, meta_data_t >::type storage_t;
        meta_data_t meta_data_(3,3,3);
        storage_t dummy(meta_data_, 0., "dummy");
        typedef arg<0, storage_t > p_dummy;

        typedef boost::mpl::vector1<p_dummy> arg_list;
        domain_type< arg_list > domain_(boost::fusion::make_vector(&dummy));

        auto comp_ = make_computation < backend<enumtype::Host, enumtype::Naive> > (
            domain_, grid_
            , make_mss(
                enumtype::execute<enumtype::forward>()
                , make_esf<functor<0> >( p_dummy() ))
            , switch_(cond
                      ,
                      case_(0
                            , make_mss(
                                enumtype::execute<enumtype::forward>()
                                , make_esf<functor<1> >( p_dummy() )) )
                      , case_(1,
                           make_mss(
                               enumtype::execute<enumtype::forward>()
                               , make_esf<functor<2> >( p_dummy() )))
                      , case_(2,
                           make_mss(
                               enumtype::execute<enumtype::forward>()
                               , make_esf<functor<3> >( p_dummy() )))
                      , case_(3,
                           make_mss(
                               enumtype::execute<enumtype::forward>()
                               , make_esf<functor<4> >( p_dummy() )))
                      , case_(4,
                           make_mss(
                               enumtype::execute<enumtype::forward>()
                               , make_esf<functor<5> >( p_dummy() )))
                      , case_(5,
                              switch_(nested_cond_
                                      , case_(2,
                                              make_mss(
                                                  enumtype::execute<enumtype::forward>()
                                                  , make_esf<functor<1000> >( p_dummy() )))
                                      , case_(1,
                                              make_mss(
                                                  enumtype::execute<enumtype::forward>()
                                                  , make_esf<functor<2000> >( p_dummy() )))
                                      , default_(
                                          make_mss(
                                              enumtype::execute<enumtype::forward>()
                                              , make_esf<functor<3000> >( p_dummy() )))
                                  )
                          )
                      , case_(6,
                              make_mss(
                                  enumtype::execute<enumtype::forward>()
                                  , make_esf<functor<6> >( p_dummy() )))
                      , default_(
                          make_mss(
                              enumtype::execute<enumtype::forward>()
                              , make_esf<functor<7> >( p_dummy() )))
                )
            ,  switch_(other_cond_
                       , case_(2,
                               make_mss(
                                   enumtype::execute<enumtype::forward>()
                                   , make_esf<functor<10> >( p_dummy() )))
                       , case_(1,
                               make_mss(
                                   enumtype::execute<enumtype::forward>()
                                   , make_esf<functor<20> >( p_dummy() )))
                       , default_(
                           make_mss(
                               enumtype::execute<enumtype::forward>()
                               , make_esf<functor<30> >( p_dummy() )))
                )
            , make_mss(
                enumtype::execute<enumtype::forward>()
                , make_esf<functor<400> >( p_dummy() ))
            );

        comp_->ready();
        comp_->steady();
        comp_->run();
        std::cout<<"\n\n\n";
        reset_conditional(cond, new_cond);
        reset_conditional(other_cond_, new_other_cond_);
        comp_->run();
        comp_->finalize();
        return 0;
    }
}//namespace test_conditional

TEST(stencil_composition, conditional_switch){
    test_conditional_switches::test();
}
