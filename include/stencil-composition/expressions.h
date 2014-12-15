#pragma once
#include<common/string_c.h>
/**@file
   @brief Expression templates definition.
   The expression templates are a method to parse at compile time the mathematical expression given
   by the user, recognizing the structure and building a syntax tree by recursively nesting
   templates.*/
namespace gridtools{

#ifdef __CUDACC__
    /**@brief Class in substitution of std::pow, not available in CUDA*/
	template <uint_t Number>
	struct products{
	    template<typename Value>
	    GT_FUNCTION
	    static Value constexpr apply(Value& v)
		{
		    return v*products<Number-1>::apply(v);
		}
	};

    /**@brief Class in substitution of std::pow, not available in CUDA*/
	template <>
	struct products<0>{
	    template<typename Value>
	    GT_FUNCTION
	    static Value constexpr apply(Value& v)
		{
		    return 1.;
		}
	};
#endif


#ifdef CXX11_ENABLED

    /** \section expressions (Expressions Definition)
        @{
        This is the base class of a binary expression, containing the instances of the two arguments.
        The expression should be a static constexpr object, instantiated once for all at the beginning of the run.
    */
    template <typename ArgType1, typename ArgType2>
    struct expr{

        /**@brief generic expression constructor*/
        GT_FUNCTION
        constexpr expr(ArgType1 const& first_operand, ArgType2 const& second_operand)
            :
            first_operand{first_operand},
            second_operand{second_operand}
            {}

	GT_FUNCTION
        constexpr expr(expr const& other):first_operand(other.first_operand),second_operand(other.second_operand){}

        ArgType1 const first_operand;
        ArgType2 const second_operand;
    private:
        /**@brief default empty constructor*/
	GT_FUNCTION
        constexpr expr(){}
    };


    template <typename ArgType1>
    struct unary_expr{
        /**@brief generic expression constructor*/
        GT_FUNCTION
        constexpr unary_expr(ArgType1 const& first_operand)
            :
            first_operand{first_operand}
            {}

	GT_FUNCTION
        constexpr unary_expr( unary_expr const& other): first_operand(other.first_operand){}

        ArgType1 const first_operand;

    private:
        /**@brief default empty constructor*/
	GT_FUNCTION
	constexpr unary_expr(){}
    };

    /**@brief Expression summing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_plus : public expr<ArgType1, ArgType2>{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_plus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

	GT_FUNCTION
        constexpr expr_plus(expr_plus const& other):super(other){};

    private:
        constexpr expr_plus(){};
	static char constexpr op[]="+";
	typedef string_c<print, op> operation;
    public:
	//currying and recursion (this gets inherited)
	using to_string = concatenate<ArgType1, concatenate<string_c<print, op>, ArgType2> >;
    };

    /**@brief Expression subrtracting two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_minus : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_minus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_minus(expr_minus const& other):super(other){}

    private:
        GT_FUNCTION
        constexpr expr_minus(){}
	static char constexpr op[]="-";
	typedef string_c<print, op> operation;
    public:
	//currying and recursion (this gets inherited)
	using to_string = concatenate<ArgType1, concatenate<string_c<print, op>, ArgType2> >;
    };

    /**@brief Expression multiplying two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_times : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_times(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_times(expr_times const& other):super(other){}
    private:
        GT_FUNCTION
        constexpr expr_times(){}
    	static char constexpr op[]="*";
    public:
	//currying and recursion (this gets inherited)
	using to_string = concatenate<ArgType1, concatenate<string_c<print, op>, ArgType2> >;
};

    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_divide : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_divide(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_divide(expr_divide const& other):super(other){}

    private:
        GT_FUNCTION
        constexpr expr_divide(){}
    	static char constexpr op[]="/";
	typedef string_c<print, op> operation;
    public:
	//currying and recursion (this gets inherited)
	using to_string = concatenate<ArgType1, concatenate<string_c<print, op>, ArgType2> >;
};

    /**@brief Expression computing the integral exponent of the first arguments
       for this expression the second argument is an integer (this might, and probably will, be relaxed if needed)
    */
    template <typename ArgType1, typename ArgType2>
    struct expr_exp : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_exp(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_exp(expr_exp const& other):super(other){}

    private:
        GT_FUNCTION
        constexpr expr_exp(){}
	static char constexpr op[]="^";
	typedef string_c<print, op> operation;
    public:
	//currying and recursion (this gets inherited)
	using to_string = concatenate<ArgType1, concatenate<string_c<print, op>, ArgType2> >;
    };


    /**@brief Expression computing the integral exponent of the first arguments
       for this expression the second argument is an integer (this might, and probably will, be relaxed if needed)
    */
    // template <int Exponent, typename ArgType1>
    // struct expr_exp : public unary_expr<ArgType1>{
    //     typedef unary_expr<ArgType1> super;
    template <typename ArgType1, int Exponent>
    struct expr_pow : public unary_expr<ArgType1 >{
        typedef unary_expr<ArgType1> super;
        GT_FUNCTION
        constexpr expr_pow(ArgType1 const& first_operand):super(first_operand){}
	static const int exponent=Exponent;

        GT_FUNCTION
        constexpr expr_pow(expr_pow const& other):super(other) {}

    private:
        GT_FUNCTION
        constexpr expr_pow(){}
    	static char constexpr op[]="^2";
	typedef string_c<print, op> operation;
    public:
	//currying and recursion (this gets inherited)
	using to_string = concatenate<  ArgType1, operation >;
};

/*@}*/

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
        // template<int Arg2, typename ArgType1>
        template<int exponent, typename ArgType1, typename boost::disable_if<typename boost::is_floating_point<ArgType1>::type, int >::type=0>
        GT_FUNCTION
        // constexpr expr_exp<Arg2, ArgType1 >    pow ( ArgType1 arg1 ){return expr_exp<Arg2, ArgType1 >(arg1);}
        constexpr expr_pow<ArgType1, exponent >    pow (ArgType1 arg1){return expr_pow<ArgType1, exponent >(arg1);}

        /** power expression*/
        // template<int Arg2, typename ArgType1>
        template<typename ArgType1>
        GT_FUNCTION
        // constexpr expr_exp<Arg2, ArgType1 >    pow ( ArgType1 arg1 ){return expr_exp<Arg2, ArgType1 >(arg1);}
        constexpr expr_exp<ArgType1, int >    pow (ArgType1 arg1, int arg2){return expr_exp<ArgType1, int >(arg1, arg2);}

	template <int Exponent, typename FloatType, typename boost::enable_if<typename boost::is_floating_point<FloatType>::type, int >::type=0>
        GT_FUNCTION
        constexpr FloatType  pow (FloatType arg1)
#ifdef __CUDACC__
	{return products<Exponent>::apply(arg1);}
#else
        {return std::pow(arg1, Exponent);}
#endif

	template<typename Left>
	GT_FUNCTION
	constexpr typename Left::super operator +(Left d1, int  offset) {return typename Left::super( offset);}

	template<typename Left>
	GT_FUNCTION
	constexpr typename Left::super operator -(Left d1, int  offset) {return typename Left::super(-offset);}

/**@}*/
    }//namespace expressions

#endif

}//namespace gridtools
