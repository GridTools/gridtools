namespace gridtools{


    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2, typename ArgType3>
    struct expr_if_then_else : public ternary_expr<ArgType1, ArgType2, ArgType3 >{
        typedef ternary_expr<ArgType1, ArgType2, ArgType3> super;
        GT_FUNCTION
        constexpr expr_if_then_else(ArgType1 const& first_operand, ArgType2 const& second_operand, ArgType3 const& third_operand):super(first_operand, second_operand, third_operand){}

        template<typename Arg1, typename Arg2, typename Arg3>
        GT_FUNCTION
        constexpr expr_if_then_else(expr_if_then_else<Arg1, Arg2, Arg3> const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_if_then_else(){}
#ifndef __CUDACC__
        static char constexpr op1[]=" ) ? (";
        static char constexpr op2[]=" ) : ( ";
        typedef string_c<print, op1> operation;
        typedef string_c<print, op2> operation2;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation,  ArgType2,  operation2, ArgType3, tokens::closed_par >;
#endif
};

    namespace expressions{

        /** if_then_else expression*/
        template<typename ArgType1, typename ArgType2, typename ArgType3>
        GT_FUNCTION
        constexpr expr_if_then_else<ArgType1, ArgType2, ArgType3>   if_then_else (ArgType1 arg1, ArgType2 arg2, ArgType3 arg3){
            return expr_if_then_else<ArgType1, ArgType2, ArgType3 >(arg1, arg2, arg3);}

        namespace evaluation{

        /** if_then_else evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2, typename ArgType3>
        GT_FUNCTION
        auto static constexpr value(IterateDomain const& it_domain
                                    , expr_if_then_else<ArgType1, ArgType2, ArgType3> const& arg)
            -> decltype(it_domain(arg.first_operand)) {
            return (it_domain(arg.first_operand))?
                arg.second_operand :
                arg.third_operand;
        }
    }//namespace evaluation
    } //namespace expressions

} //namespace gridtools
