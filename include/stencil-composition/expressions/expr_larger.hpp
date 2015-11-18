namespace gridtools{

    /**@brief Expression for comparison*/
    template <typename ArgType1, typename ArgType2>
    struct expr_larger : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_larger(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        template<typename Arg1, typename Arg2>
        GT_FUNCTION
        constexpr expr_larger(expr_larger<Arg1, Arg2> const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_larger(){}
#ifndef __CUDACC__

        static char constexpr op[]=" > ";
        using operation = string_c<print, op>;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    namespace expressions{

        /** larger expression*/
        template<typename ArgType1, typename ArgType2 ,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_larger<ArgType1, ArgType2>   operator > (ArgType1 arg1, ArgType2 arg2){
            return expr_larger<ArgType1, ArgType2 >(arg1, arg2);}


    namespace evaluation{
        /** larger of evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        bool static constexpr value(IterateDomain const& it_domain
                                    , expr_larger<ArgType1, ArgType2> const& arg){
            return it_domain(arg.first_operand) > it_domain(arg.second_operand);}
    }//namespace evaluation
    } //namespace expressions

} //namespace gridtools
