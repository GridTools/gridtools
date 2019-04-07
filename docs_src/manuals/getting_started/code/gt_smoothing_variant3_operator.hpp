struct smoothing_function_3 {
    GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(phi), GT_IN_ACCESSOR(lap, extent<-1, 1, -1, 1>), GT_INOUT_ACCESSOR(out));

    constexpr static double alpha = 0.5;

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
