struct smoothing_function_3 {
    using phi = in_accessor<0>;
    using lap = in_accessor<1, extent<-1, 1, -1, 1>>;
    using out = inout_accessor<2>;

    using param_list = make_param_list<phi, lap, out>;

    constexpr static double alpha = 0.5;

    constexpr static auto i = dimension<1>();
    constexpr static auto j = dimension<2>();
    constexpr static auto k = dimension<3>();

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, lower_domain) {
        eval(out(i, j, k)) = eval(phi(i, j, k)) - alpha *                //
                                                      call<lap_function> //
                                                      ::with(eval, lap());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, upper_domain) {
        eval(out(i, j, k)) = eval(phi(i, j, k));
    }
};
