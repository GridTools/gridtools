namespace gridtools {

    /**@brief Expression multiplying two arguments*/
    template < typename ArgType1, typename ArgType2 >
    struct expr_times : public expr< ArgType1, ArgType2 > {
        typedef expr< ArgType1, ArgType2 > super;
        GT_FUNCTION
        constexpr expr_times(ArgType1 const &first_operand, ArgType2 const &second_operand)
            : super(first_operand, second_operand) {}

        template < typename Arg1, typename Arg2 >
        GT_FUNCTION constexpr expr_times(expr_times< Arg1, Arg2 > const &other)
            : super(other) {}
#ifndef __CUDACC__
      private:
#endif
        GT_FUNCTION
        constexpr expr_times() {}
#ifndef __CUDACC__
        static char constexpr op[] = " * ";
        typedef string_c< print, op > operation;

      public:
        // currying and recursion (this gets inherited)
        using to_string = concatenate< tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template < typename Arg >
    struct is_binary_expr;

    template < typename Arg1, typename Arg2 >
    struct no_expr_nor_accessor_types;

    template < typename ArgType1, typename ArgType2 >
    struct is_binary_expr< expr_times< ArgType1, ArgType2 > > : boost::mpl::true_ {};

    namespace expressions {

        /** multiply expression*/
        template < typename ArgType1,
            typename ArgType2,
            typename boost::disable_if< no_expr_nor_accessor_types< ArgType1, ArgType2 >, int >::type = 0 >
        GT_FUNCTION constexpr expr_times< ArgType1, ArgType2 > operator*(ArgType1 arg1, ArgType2 arg2) {
            return expr_times< ArgType1, ArgType2 >(arg1, arg2);
        }

        namespace evaluation {

            /** multiplication evaluation*/
            template < typename IterateDomain, typename ArgType1, typename ArgType2 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_times< ArgType1, ArgType2 > const &arg)
                -> decltype(it_domain(arg.first_operand) * it_domain(arg.second_operand)) {
                return it_domain(arg.first_operand) * it_domain(arg.second_operand);
            }

            /** multiply with scalar evaluation*/
            template < typename IterateDomain,
                typename ArgType1,
                typename FloatType,
                typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_times< ArgType1, FloatType > const &arg)
                -> decltype(it_domain(arg.first_operand) * arg.second_operand) {
                return it_domain(arg.first_operand) * arg.second_operand;
            }

            /** multiply with scalar evaluation*/
            template < typename IterateDomain,
                typename FloatType,
                typename ArgType2,
                typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_times< FloatType, ArgType2 > const &arg)
                -> decltype(arg.first_operand * it_domain(arg.second_operand) ) {
                return arg.first_operand * it_domain(arg.second_operand);
            }

            // automatic differentiation
            /** times derivative evaluation*/
            template < typename IterateDomain, typename ArgType1, typename ArgType2 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_derivative< expr_times< ArgType1, ArgType2 > > const &arg)
                -> decltype(it_domain(expr_derivative< ArgType1 >(arg.first_operand)) * it_domain(arg.second_operand) +
                            it_domain(arg.first_operand) * it_domain(expr_derivative< ArgType2 >(arg.second_operand))) {
                return it_domain(expr_derivative< ArgType1 >(arg.first_operand) * arg.second_operand +
                                 arg.first_operand * expr_derivative< ArgType2 >(arg.second_operand));
            }

            /** multiply with scalar evaluation*/
            template < typename IterateDomain,
                typename ArgType1,
                typename FloatType,
                typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_derivative< expr_times< ArgType1, FloatType > > const &arg)
                -> decltype(it_domain(expr_derivative< ArgType1 >(arg.first_operand)) * arg.second_operand) {
                return it_domain(expr_derivative< ArgType1 >(arg.first_operand) * arg.second_operand);
            }

            /** multiply a scalar evaluation*/
            template < typename IterateDomain,
                typename FloatType,
                typename ArgType2,
                typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_derivative< expr_times< FloatType, ArgType2 > > const &arg)
                -> decltype(it_domain(expr_derivative< ArgType2 >(arg.second_operand)) * arg.first_operand) {
                return it_domain(expr_derivative< ArgType2 >(arg.second_operand) * arg.first_operand);
            }

        } // namespace evaluation
    } // namespace expressions

} // namespace gridtools
