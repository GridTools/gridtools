namespace gridtools{


    /**@brief Expression subrtracting two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_minus : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;

        GT_FUNCTION
        constexpr expr_minus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        template<typename Arg1, typename Arg2>
        GT_FUNCTION
        constexpr expr_minus(expr_minus<Arg1, Arg2> const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_minus(){}

#ifndef __CUDACC__
        static char constexpr op[]=" - ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template < typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_minus<ArgType1, ArgType2> > : boost::mpl::true_ {
    };

    namespace expressions{
        /** minus expression*/
        template<typename ArgType1, typename ArgType2,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_minus<ArgType1, ArgType2 > operator - (ArgType1 arg1, ArgType2 arg2){
            return expr_minus<ArgType1, ArgType2 >(arg1, arg2);}

    namespace evaluation{

        /** minus evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto static constexpr value(IterateDomain const& it_domain
                                    , expr_minus<ArgType1, ArgType2> const& arg)
            -> decltype(it_domain(arg.first_operand) - it_domain(arg.second_operand)) {
            return it_domain(arg.first_operand) - it_domain(arg.second_operand);}


        /** subtract with scalar evaluation*/
        template <typename IterateDomain, typename ArgType1, typename FloatType
                  , typename boost::enable_if<
                        typename boost::is_floating_point<FloatType>::type
                        , int >::type=0
                  >
        GT_FUNCTION
        auto static constexpr value_scalar(IterateDomain const& it_domain
                                           , expr_minus<ArgType1, FloatType> const& arg)
            -> decltype(it_domain(arg.first_operand) - arg.second_operand) {
            return it_domain(arg.first_operand) - arg.second_operand;}


        //automatic differentiation
        /** plus derivative evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto static constexpr value(IterateDomain const& it_domain
                                    , expr_derivative<expr_minus<ArgType1, ArgType2> > const& arg)
            -> decltype(it_domain(expr_derivative<ArgType1>(arg.first_operand)) - it_domain( expr_derivative<ArgType2>(arg.second_operand) ) ) {
            return it_domain(expr_derivative<ArgType1>(arg.first_operand)) - it_domain( expr_derivative<ArgType2>(arg.second_operand) );}

        /** sum with scalar evaluation*/
        template <typename IterateDomain, typename ArgType1, typename FloatType
                  , typename boost::enable_if<
                        typename boost::is_floating_point<FloatType>::type
                        , int >::type=0
                  >
        GT_FUNCTION
        auto static constexpr value_scalar(IterateDomain const& it_domain
                                           , expr_derivative<expr_minus<ArgType1, FloatType> > const& arg)
            -> decltype(it_domain(expr_derivative<ArgType1>(arg.first_operand) ) ) {
            return it_domain(expr_derivative<ArgType1>(arg.first_operand) ) ;}

    }//namespace evaluation
    }//namespace expressions

} //namespace gridtools
