#pragma once
namespace gridtools {

    /**@brief decorates the user function with a defaiult interval, in case no interval was specified by the user

       A SFINAE mechanism detects wether the user gave the vertical interval as input to the Do method,
       and when this is not the case it wraps the functor inside this decoratpr, passing the whole axis to it.
       The first and last points are removed from the axis (the GridTools API works with exclusive intervals). So
       the functor_decorator spans the whole domain embedded in the vertical axis.
       \tparam F the user functor
       \tparam Axis the vertical axis
     */
    template < typename F, typename Axis >
    struct functor_decorator {
        static const constexpr int_t to_offset = Axis::ToLevel::Offset::value;
        static const constexpr uint_t to_splitter = Axis::ToLevel::Splitter::value;
        static const constexpr int_t from_offset = Axis::FromLevel::Offset::value;
        static const constexpr uint_t from_splitter = Axis::FromLevel::Splitter::value;

        // NOTE: the offsets cannot be 0
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
