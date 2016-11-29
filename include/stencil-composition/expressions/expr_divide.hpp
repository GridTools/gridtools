namespace gridtools {

    /**@brief Expression dividing two arguments*/
    template < typename ArgType1, typename ArgType2 >
    struct expr_divide : public binary_expr< ArgType1, ArgType2 > {
        typedef binary_expr< ArgType1, ArgType2 > super;
        GT_FUNCTION
        constexpr expr_divide(ArgType1 const &first_operand, ArgType2 const &second_operand)
            : super(first_operand, second_operand) {}

        template < typename Arg1, typename Arg2 >
        GT_FUNCTION constexpr expr_divide(expr_divide< Arg1, Arg2 > const &other)
            : super(other) {}

#ifndef __CUDACC__
      private:
#endif
        GT_FUNCTION
        constexpr expr_divide() {}
#ifndef __CUDACC__
        static char constexpr op[] = " / ";
        typedef string_c< print, op > operation;

      public:
        // currying and recursion (this gets inherited)
        using to_string = concatenate< tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template < typename ArgType1, typename ArgType2 >
    struct is_binary_expr< expr_divide< ArgType1, ArgType2 > > : boost::mpl::true_ {};

    namespace expressions {

        /** divide expression*/
        template < typename ArgType1,
            typename ArgType2,
            typename boost::disable_if< no_expr_nor_accessor_types< ArgType1, ArgType2 >, int >::type = 0 >
        GT_FUNCTION constexpr expr_divide< ArgType1, ArgType2 > operator/(ArgType1 arg1, ArgType2 arg2) {
            return expr_divide< ArgType1, ArgType2 >(arg1, arg2);
        }

        namespace evaluation {

            /** division evaluation*/
            template < typename IterateDomain, typename ArgType1, typename ArgType2 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_divide< ArgType1, ArgType2 > const &arg)
                -> decltype(it_domain(arg.first_operand) / it_domain(arg.second_operand)) {
                return it_domain(arg.first_operand) / it_domain(arg.second_operand);
            }

            /** divide with scalar evaluation*/
            template < typename IterateDomain,
                typename ArgType1,
                typename FloatType,
                typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_divide< ArgType1, FloatType > const &arg)
                -> decltype(it_domain(arg.first_operand) / arg.second_operand) {
                return it_domain(arg.first_operand) / arg.second_operand;
            }

            /** divide a scalar evaluation (non commutative)*/
            template < typename IterateDomain,
                typename FloatType,
                typename ArgType2,
                typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_divide< FloatType, ArgType2 > const &arg)
                -> decltype(arg.first_operand / it_domain(arg.second_operand)) {
                return arg.second_operand / it_domain(arg.second_operand);
            }

            // automatic differentiation
            /** divison derivative evaluation*/
            template < typename IterateDomain,
                typename ArgType1,
                typename ArgType2,
                typename boost::enable_if< typename boost::mpl::or_< typename is_expr< ArgType2 >::type,
                                               typename is_accessor< ArgType2 >::type >,
                    int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_derivative< expr_divide< ArgType1, ArgType2 > > const &arg)
                -> decltype(it_domain((expr_derivative< ArgType1 >(arg.first_operand) * arg.second_operand -
                                          arg.first_operand * expr_derivative< ArgType2 >(arg.second_operand)) /
                                      (::gridtools::expressions::pow< 2 >(arg.first_operand) +
                                          ::gridtools::expressions::pow< 2 >(arg.second_operand)))) {
                return it_domain((expr_derivative< ArgType1 >(arg.first_operand) * arg.second_operand -
                                     arg.first_operand * expr_derivative< ArgType2 >(arg.second_operand)) /
                                 (::gridtools::expressions::pow< 2 >(arg.first_operand) +
                                     ::gridtools::expressions::pow< 2 >(arg.second_operand)));
            }

            /** divide with scalar evaluation*/
            template < typename IterateDomain,
                typename ArgType1,
                typename FloatType,
                typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_derivative< expr_divide< ArgType1, FloatType > > const &arg)
                -> decltype(it_domain(
                    expr_derivative< ArgType1 >(arg.first_operand.first_operand) / arg.first_operand.second_operand)) {
                return it_domain(
                    expr_derivative< ArgType1 >(arg.first_operand.first_operand) / arg.first_operand.second_operand);
            }

            /** divide a scalar evaluation*/
            template < typename IterateDomain,
                typename FloatType,
                typename ArgType2,
                typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
            GT_FUNCTION auto static constexpr value(
                IterateDomain const &it_domain, expr_derivative< expr_divide< ArgType2, FloatType > > const &arg)
                -> decltype(it_domain(
                    (arg.first_operand * expr_derivative< ArgType2 >(arg.second_operand)) /
                    (gt_pow< 2 >::apply(arg.first_operand) + ::gridtools::expressions::pow< 2 >(arg.second_operand)))) {
                return -it_domain(
                    (arg.first_operand * expr_derivative< ArgType2 >(arg.second_operand)) /
                    (gt_pow< 2 >::apply(arg.first_operand) + ::gridtools::expressions::pow< 2 >(arg.second_operand)));
            }

        } // namespace evaluation
    }     // namespace expressions

} // namespace gridtools
