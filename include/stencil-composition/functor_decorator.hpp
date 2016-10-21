#pragma once
namespace gridtools {

    template < typename F, typename Axis >
    struct functor_decorator {
        static constexpr int_t to_offset = Axis::ToLevel::Offset::value;
        static constexpr uint_t to_splitter = Axis::ToLevel::Splitter::value;

        typedef gridtools::interval< typename Axis::FromLevel,
            level< to_splitter, (to_offset != 1) ? to_offset - 1 : to_offset - 2 > > default_interval;

        typedef F type;

        typedef typename F::arg_list arg_list;
        template < typename Eval >
        GT_FUNCTION static void Do(Eval const &eval_, default_interval) {
            F::Do(eval_);
        }
    };
} // namespace gridtools
