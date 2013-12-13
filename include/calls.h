namespace gridtools {
    template <int im, int ip, int jm, int jp, int kp, int km>
    struct _rect {};

    template <typename t_functor>
    struct calls {

        template <typename t_arg0>
        struct with {
            template <typename rectangle>
            struct at {
                typedef t_arg0 arg0_type;
                typedef t_functor functor_type;
                typedef rectangle rect_type;
            };
        };

    };
} // namespace gridtools

