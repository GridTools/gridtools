#pragma once
namespace gridtools {

    template < typename F, typename Axis >
    struct functor_decorator {
        static const constexpr int_t to_offset = Axis::ToLevel::Offset::value;
        static const constexpr uint_t to_splitter = Axis::ToLevel::Splitter::value;
        static const constexpr int_t from_offset = Axis::FromLevel::Offset::value;
        static const constexpr uint_t from_splitter = Axis::FromLevel::Splitter::value;

        typedef gridtools::interval< level< from_splitter, from_offset + 1 >,
            level< to_splitter, (to_offset != 1) ? to_offset - 1 : to_offset - 2 > > default_interval;

        typedef F type;

        typedef typename F::arg_list arg_list;
        template < typename Eval >
        GT_FUNCTION static void Do(Eval const &eval_, default_interval) {
            F::Do(eval_);
        }
    };
    template < typename T >
    struct is_functor_decorator : boost::mpl::false_ {};

    template < typename T, typename A >
    struct is_functor_decorator< functor_decorator< T, A > > : boost::mpl::true_ {};
} // namespace gridtools
