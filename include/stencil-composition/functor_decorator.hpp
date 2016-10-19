#pragma once
namespace gridtools{

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > default_interval;

    template <typename F>
    struct functor_decorator {

        typedef typename F::arg_list arg_list;
        template <typename Eval>
        GT_FUNCTION
        static void Do(Eval const& eval_, default_interval){
            F::Do(eval_);
        }

        // adding an ExtraCrap useless argument in order to trick the has_do metafunctions,
        // which make sure there is only a single Do method
        template <typename Eval, typename Interval, typename ExtraCrap >
        GT_FUNCTION
        static void Do(Eval const& eval_, Interval, ExtraCrap=default_interval()){
            F::Do(eval_, Interval() );
        }
    };
}//namespace gridtools
