namespace gridtools {

    /**@brief Expression computing the integral exponent of the first arguments
       for this expression the second argument is an integer (this might, and probably will, be relaxed if needed)
    */
    template < typename ArgType1, typename ArgType2 >
    struct expr_exp : public expr< ArgType1, ArgType2 > {
        typedef expr< ArgType1, ArgType2 > super;
        GT_FUNCTION
        constexpr expr_exp(ArgType1 const &first_operand, ArgType2 const &second_operand)
            : super(first_operand, second_operand) {}

        template < typename Arg1, typename Arg2 >
        GT_FUNCTION constexpr expr_exp(expr_exp< Arg1, Arg2 > const &other)
            : super(other) {}

#ifndef __CUDACC__
      private:
#endif
        GT_FUNCTION
        constexpr expr_exp() {}
#ifndef __CUDACC__
        static char constexpr op[] = " ^ ";
        typedef string_c< print, op > operation;

      public:
        // currying and recursion (this gets inherited)
        using to_string = concatenate< tokens::open_par,
            ArgType1,
            tokens::closed_par,
            operation,
            tokens::open_par,
            ArgType2,
            tokens::closed_par >;
#endif
    };

    namespace expressions {

        /** power expression*/
        template < typename ArgType1 >
        GT_FUNCTION constexpr expr_exp< ArgType1, int > pow(ArgType1 arg1, int arg2) {
            return expr_exp< ArgType1, int >(arg1, arg2);
        }
    } // namespace expressions

    namespace evaluation {

        /** power of scalar evaluation*/
        template < typename IterateDomain,
            typename FloatType,
            typename IntType,
            typename boost::enable_if< typename boost::is_floating_point< FloatType >::type, int >::type = 0,
            typename boost::enable_if< typename boost::is_integral< IntType >::type, int >::type = 0 >
        GT_FUNCTION static auto constexpr value(IterateDomain const & /*it_domain*/
            ,
            expr_exp< FloatType, IntType > const &arg) -> decltype(std::pow(arg.first_operand, arg.second_operand)) {
            return gt_pow< 2 >::apply(arg.first_operand);

        } // namespace evaluation
    }

} // namespace gridtools
