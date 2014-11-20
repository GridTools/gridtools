/**@file
   @brief Expression templates definition.
   The expression templates are a method to parse at compile time the mathematical expression given
   by the user, recognizing the structure and building a syntax tree by recursively nesting
   templates.*/
namespace gridtools{

    /** \section expressions (Expressions Definition)
        @{
        This is the base class of a binary expression, containing the instances of the two argument.
        The expression should be a static constexpr object, instantiated once for all at the beginning of the run.
    */
    template <typename ArgType1, typename ArgType2>
    struct expr{

        /**@brief generic expression constructor*/
        GT_FUNCTION
        constexpr expr(ArgType1 const& first_operand, ArgType2 const& second_operand)
            :
#if( (!defined(CXX11_ENABLED)))
            first_operand(first_operand),
            second_operand(second_operand)
#else
            first_operand{first_operand},
            second_operand{second_operand}
#endif
            {}

        /**@brief default empty constructor*/
        constexpr expr(){}
        ArgType1 const first_operand;
        ArgType2 const second_operand;
    };

    /**@brief Expression summing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_plus : public expr<ArgType1, ArgType2>{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_plus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
        constexpr expr_plus(){};
    };

    /**@brief Expression subrtracting two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_minus : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_minus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
        constexpr expr_minus(){}
    };

    /**@brief Expression multiplying two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_times : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_times(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
        constexpr expr_times(){}
    };

    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_divide : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_divide(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
        constexpr expr_divide(){}
    };

    /**@brief Expression computing the integral exponent of the first arguments
       for this expression the second argument is an integer (this might, and probably will, be relaxed if needed)
    */
    template <typename ArgType1, typename ArgType2>
    struct expr_exp : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_exp(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
        constexpr expr_exp(){}
    };
/*@}*/

#ifdef CXX11_ENABLED
    /**@brief Overloaded operators
       The algebraic operators are overloaded in order to deal with expressions. To enable these operators the user has to use the namespace expressions.*/
    namespace expressions{
/**\section operator (Operators Overloaded)
   @{*/
        /** sum expression*/
        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_plus<ArgType1, ArgType2 >  operator + (ArgType1 arg1, ArgType2 arg2){return expr_plus<ArgType1, ArgType2 >(arg1, arg2);}

        /** minus expression*/
        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_minus<ArgType1, ArgType2 > operator - (ArgType1 arg1, ArgType2 arg2){return expr_minus<ArgType1, ArgType2 >(arg1, arg2);}

        /** multiply expression*/
        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_times<ArgType1, ArgType2 > operator * (ArgType1 arg1, ArgType2 arg2){return expr_times<ArgType1, ArgType2 >(arg1, arg2);}

        /** divide expression*/
        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_divide<ArgType1, ArgType2 > operator / (ArgType1 arg1, ArgType2 arg2){return expr_divide<ArgType1, ArgType2 >(arg1, arg2);}

        /** power expression*/
        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_exp<ArgType1, ArgType2 >    operator ^ (ArgType1 arg1, ArgType2 arg2){return expr_exp<ArgType1, ArgType2 >(arg1, arg2);}
/**@}*/
    }//namespace expressions

#endif

}//namespace gridtools
