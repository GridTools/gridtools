namespace gridtools {

    /**@brief Expression enabling the direct access to the storage.

       The offsets only (without the index) identify the memory address to be used
    */
    template < typename ArgType1 >
    struct expr_direct_access : public unary_expr< ArgType1 > {
        typedef unary_expr< ArgType1 > super;
        GT_FUNCTION
        constexpr expr_direct_access(ArgType1 const &first_operand) : super(first_operand) {}

        template < typename Arg1 >
        GT_FUNCTION constexpr expr_direct_access(expr_direct_access< Arg1 > const &other)
            : super(other) {}

#ifndef __CUDACC__
      private:
#endif
        GT_FUNCTION
        constexpr expr_direct_access() {}
#ifndef __CUDACC__
        static char constexpr op[] = " !";
        typedef string_c< print, op > operation;

      public:
        // currying and recursion (this gets inherited)
        using to_string = concatenate< operation, tokens::open_par, ArgType1, tokens::closed_par >;
#endif
    };

    template < typename ArgType1 >
    struct is_unary_expr< expr_direct_access< ArgType1 > > : boost::mpl::true_ {};

    namespace expressions {

        /** direct access expression*/
        template < typename ArgType1,
            typename boost::disable_if< no_expr_nor_accessor_types< ArgType1, int >, int >::type = 0 >
        GT_FUNCTION constexpr expr_direct_access< ArgType1 > operator!(ArgType1 arg1) {
            return expr_direct_access< ArgType1 >(arg1);
        }
    } // namespace expressions

} // namespace gridtools
