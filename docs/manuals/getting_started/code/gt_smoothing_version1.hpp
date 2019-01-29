struct smoothing_function_1 {
    using phi = in_accessor<0>;
    using laplap = in_accessor<1>;
    using out = inout_accessor<2>;

    using arg_list = boost::mpl::vector<phi, laplap, out>;

    constexpr static double alpha = 0.5;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, lower_domain) {
        eval(out(i, j, k)) = eval(phi(i, j, k)) //
                             - alpha * eval(laplap(i, j, k));
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, upper_domain) {
        eval(out(i, j, k)) = eval(phi(i, j, k));
    }
};
