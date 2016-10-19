#pragma once
namespace gridtools {

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > default_interval;

    template < typename F >
    struct functor_decorator {
        typedef F type;

        typedef typename F::arg_list arg_list;
        template < typename Eval >
        GT_FUNCTION static void Do(Eval const &eval_, default_interval) {
            F::Do(eval_);
        }
    };
} // namespace gridtools
