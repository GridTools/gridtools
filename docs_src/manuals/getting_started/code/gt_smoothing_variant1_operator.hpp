struct smoothing_function_1 {
    GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(phi), GT_IN_ACCESSOR(laplap), GT_INOUT_ACCESSOR(out));

    constexpr static double alpha = 0.5;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, lower_domain) {
        eval(out(i, j, k)) = eval(phi(i, j, k)) //
                             - alpha * eval(laplap(i, j, k));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, upper_domain) {
        eval(out(i, j, k)) = eval(phi(i, j, k));
    }
};
