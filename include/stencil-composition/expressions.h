#pragma once
#include<common/defs.h>
#include<common/string_c.h>
#include <common/gt_math.h>

/**@file
   @brief Expression templates definition.
   The expression templates are a method to parse at compile time the mathematical expression given
   by the user, recognizing the structure and building a syntax tree by recursively nesting
   templates.*/
namespace gridtools{

#ifdef CXX11_ENABLED

    struct tokens{
        static char constexpr par_o[]="(";
        static char constexpr par_c[]=")";
        using  open_par = string_c<print, par_o>;
        using  closed_par = string_c<print, par_c>;
    };
    /** \section expressions Expressions Definition
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
#ifndef __CUDACC__
    private:
#endif
      /**@brief default empty constructor*/
        GT_FUNCTION
        constexpr expr(){}
    };

    template <typename Arg>
    struct is_binary_expr : boost::mpl::false_ {};


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

#ifndef __CUDACC__
    private:
#endif
      /**@brief default empty constructor*/
        GT_FUNCTION
        constexpr unary_expr(){}
    };

        template <typename ArgType1, typename ArgType2, typename ArgType3>
    struct ternary_expr{

        /**@brief generic expression constructor*/
        GT_FUNCTION
        constexpr ternary_expr(ArgType1 const& first_operand, ArgType2 const& second_operand, ArgType3 const& third_operand)
            :
            first_operand{first_operand},
            second_operand{second_operand},
            third_operand{third_operand}
            {}

        GT_FUNCTION
        constexpr ternary_expr(ternary_expr const& other) :
            first_operand(other.first_operand),
            second_operand(other.second_operand),
            third_operand(other.third_operand){}

        ArgType1 const first_operand;
        ArgType2 const second_operand;
        ArgType3 const third_operand;
#ifndef __CUDACC__
    private:
#endif
        /**@brief default empty constructor*/
        GT_FUNCTION
        constexpr ternary_expr(){}
    };

    template <typename Arg>
    struct is_unary_expr : boost::mpl::false_ {};

    template <typename Arg>
    using is_expr=typename boost::mpl::or_<is_binary_expr<Arg>, is_unary_expr<Arg> >::type;

    /**@brief Expression summing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_plus : public expr<ArgType1, ArgType2>{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_plus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_plus(expr_plus const& other):super(other){};

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_plus(){};
#ifndef __CUDACC__
        static char constexpr op[]=" + ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par>;
#endif
    };

    template < typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_plus<ArgType1, ArgType2> > : boost::mpl::true_ {
    };


    /**@brief Expression subrtracting two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_minus : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;

        GT_FUNCTION
        constexpr expr_minus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_minus(expr_minus const& other):super(other){}

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

    /**@brief Expression multiplying two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_times : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_times(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_times(expr_times const& other):super(other){}
#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_times(){}
#ifndef __CUDACC__
        static char constexpr op[]=" * ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template < typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_times<ArgType1, ArgType2> > : boost::mpl::true_ {
    };

    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_divide : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_divide(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_divide(expr_divide const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_divide(){}
#ifndef __CUDACC__
        static char constexpr op[]=" / ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template < typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_divide<ArgType1, ArgType2> > : boost::mpl::true_ {
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

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_exp(){}
#ifndef __CUDACC__
        static char constexpr op[]=" ^ ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, tokens::closed_par, operation, tokens::open_par, ArgType2, tokens::closed_par >;
#endif
    };

    template <typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_exp<ArgType1, ArgType2> > : boost::mpl::true_ {};

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

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_pow(){}
#ifndef __CUDACC__
        static char constexpr op[]="^2 ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par,  ArgType1, tokens::closed_par, operation >;
#endif
    };

    template <typename ArgType1, int Exponent>
    struct is_unary_expr<expr_pow<ArgType1, Exponent> > : boost::mpl::true_ {};


    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_larger : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_larger(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_larger(expr_larger const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_larger(){}
#ifndef __CUDACC__
        static char constexpr op[]=" > ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template <typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_larger<ArgType1, ArgType2> > : boost::mpl::true_ {};

        /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_larger_equal : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_larger_equal(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_larger_equal(expr_larger_equal const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_larger_equal(){}
#ifndef __CUDACC__
        static char constexpr op[]=" > ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template <typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_larger_equal<ArgType1, ArgType2> > : boost::mpl::true_ {};

    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_smaller : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_smaller(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_smaller(expr_smaller const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_smaller(){}
#ifndef __CUDACC__
        static char constexpr op[]=" < ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template <typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_smaller<ArgType1, ArgType2> > : boost::mpl::true_ {};

        /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_smaller_equal : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_smaller_equal(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_smaller_equal(expr_smaller_equal const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_smaller_equal(){}
#ifndef __CUDACC__
        static char constexpr op[]=" <= ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template <typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_smaller_equal<ArgType1, ArgType2> > : boost::mpl::true_ {};


    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_and : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_and(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_and(expr_and const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_and(){}
#ifndef __CUDACC__
        static char constexpr op[]=" && ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template <typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_and<ArgType1, ArgType2> > : boost::mpl::true_ {};

    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2>
    struct expr_or : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_or(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}

        GT_FUNCTION
        constexpr expr_or(expr_or const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_or(){}
#ifndef __CUDACC__
        static char constexpr op[]=" || ";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<tokens::open_par, ArgType1, operation, ArgType2, tokens::closed_par >;
#endif
    };

    template <typename ArgType1, typename ArgType2>
    struct is_binary_expr<expr_or<ArgType1, ArgType2> > : boost::mpl::true_ {};



    /**@brief Expression dividing two arguments*/
    template <typename ArgType1, typename ArgType2, typename ArgType3>
    struct expr_if_then_else : public ternary_expr<ArgType1, ArgType2, ArgType3 >{
        typedef ternary_expr<ArgType1, ArgType2, ArgType3> super;
        GT_FUNCTION
        constexpr expr_if_then_else(ArgType1 const& first_operand, ArgType2 const& second_operand, ArgType3 const& third_operand):super(first_operand, second_operand, third_operand){}

        GT_FUNCTION
        constexpr expr_if_then_else(expr_if_then_else const& other):super(other){}

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




    /**@brief Expression enabling the direct access to the storage.

       The offsets only (without the index) identify the memory address to be used
    */
    template <typename ArgType1>
    struct expr_direct_access : public unary_expr<ArgType1 >{
        typedef unary_expr<ArgType1> super;
        GT_FUNCTION
        constexpr expr_direct_access(ArgType1 const& first_operand):super(first_operand){}

        GT_FUNCTION
        constexpr expr_direct_access(expr_direct_access const& other):super(other) {}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_direct_access(){}
#ifndef __CUDACC__
        static char constexpr op[]=" !";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate< operation, tokens::open_par, ArgType1, tokens::closed_par >;
#endif
    };

    template <typename ArgType1>
    struct is_unary_expr<expr_direct_access<ArgType1> > : boost::mpl::true_ {};

/*@}*/
        template <typename Arg>
        struct is_accessor;

    /**@brief Overloaded operators
       The algebraic operators are overloaded in order to deal with expressions. To enable these operators the user has to use the namespace expressions.*/
    namespace expressions{
/**\section operator (Operators Overloaded)
   @{*/

        template<typename Arg1, typename Arg2 >
        using both_arithmetic_types = typename boost::mpl::and_<boost::is_arithmetic<Arg1>, boost::is_arithmetic<Arg2> >::type;

        template<typename Arg1, typename Arg2 >
        using no_expr_types = typename boost::mpl::not_<typename boost::mpl::or_<is_expr<Arg1>, is_expr<Arg2> >::type>::type ;

        template<typename Arg1, typename Arg2 >
        using no_accessor_types = typename boost::mpl::not_<typename boost::mpl::or_<is_accessor<Arg1>, is_accessor<Arg2> >::type>::type ;

        template<typename Arg1, typename Arg2 >
        using no_expr_nor_accessor_types = typename boost::mpl::and_<no_accessor_types<Arg1, Arg2>, no_expr_types<Arg1, Arg2> >::type ;

        /** sum expression*/
        template<typename ArgType1, typename ArgType2 ,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_plus<ArgType1, ArgType2 >  operator + (ArgType1 arg1, ArgType2 arg2){
            return expr_plus<ArgType1, ArgType2 >(arg1, arg2);}

        /** minus expression*/
        template<typename ArgType1, typename ArgType2,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_minus<ArgType1, ArgType2 > operator - (ArgType1 arg1, ArgType2 arg2){
            return expr_minus<ArgType1, ArgType2 >(arg1, arg2);}

        /** multiply expression*/
        template<typename ArgType1, typename ArgType2,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_times<ArgType1, ArgType2 > operator * (ArgType1 arg1, ArgType2 arg2){
            return expr_times<ArgType1, ArgType2 >(arg1, arg2);}

        /** divide expression*/
        template<typename ArgType1, typename ArgType2,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_divide<ArgType1, ArgType2 > operator / (ArgType1 arg1, ArgType2 arg2){
            return expr_divide<ArgType1, ArgType2 >(arg1, arg2);}

        /** power expression*/
        template<int exponent, typename ArgType1,
                 typename boost::disable_if<typename boost::is_floating_point<ArgType1>::type, int >::type=0>
        GT_FUNCTION
        constexpr expr_pow<ArgType1, exponent >    pow (ArgType1 arg1){
            return expr_pow<ArgType1, exponent >(arg1);}

        /** power expression*/
        template<typename ArgType1>
        GT_FUNCTION
        constexpr expr_exp<ArgType1, int >    pow (ArgType1 arg1, int arg2){
            return expr_exp<ArgType1, int >(arg1, arg2);}

        /** larger expression*/
        template<typename ArgType1, typename ArgType2 ,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_larger<ArgType1, ArgType2>   operator > (ArgType1 arg1, ArgType2 arg2){
            return expr_larger<ArgType1, ArgType2 >(arg1, arg2);}

        /** larger or equal expression*/
        template<typename ArgType1, typename ArgType2 ,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_larger_equal<ArgType1, ArgType2>   operator >= (ArgType1 arg1, ArgType2 arg2){
            return expr_larger_equal<ArgType1, ArgType2 >(arg1, arg2);}

        /** smaller expression*/
        template<typename ArgType1, typename ArgType2,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_smaller<ArgType1, ArgType2 >   operator < (ArgType1 arg1, ArgType2 arg2){
            return expr_smaller<ArgType1, ArgType2 >(arg1, arg2);}

        /** smaller expression*/
        template<typename ArgType1, typename ArgType2,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_smaller_equal<ArgType1, ArgType2 >   operator <= (ArgType1 arg1, ArgType2 arg2){
            return expr_smaller_equal<ArgType1, ArgType2 >(arg1, arg2);}

        /** and expression*/
        template<typename ArgType1, typename ArgType2 ,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_and<ArgType1, ArgType2 >   operator && (ArgType1 arg1, ArgType2 arg2){
            return expr_and<ArgType1, ArgType2 >(arg1, arg2);}

        /** and expression*/
        template<typename ArgType1, typename ArgType2 ,
                 typename boost::disable_if<
                     no_expr_nor_accessor_types< ArgType1, ArgType2 >
                     , int >::type=0 >
        GT_FUNCTION
        constexpr expr_or<ArgType1, ArgType2 >   operator || (ArgType1 arg1, ArgType2 arg2){
            return expr_or<ArgType1, ArgType2 >(arg1, arg2);}

        /** if_then_else expression*/
        template<typename ArgType1, typename ArgType2, typename ArgType3>
        GT_FUNCTION
        constexpr expr_if_then_else<ArgType1, ArgType2, ArgType3>   if_then_else (ArgType1 arg1, ArgType2 arg2, ArgType3 arg3){
            return expr_if_then_else<ArgType1, ArgType2, ArgType3 >(arg1, arg2, arg3);}

        /** direct access expression*/
        template<typename ArgType1>
        GT_FUNCTION
        constexpr expr_direct_access<ArgType1>    operator ! (ArgType1 arg1){
            return expr_direct_access<ArgType1>(arg1);}

        template <int Exponent, typename FloatType,
                  typename boost::enable_if<
                      typename boost::is_floating_point<FloatType>::type, int >::type=0
                  >
        GT_FUNCTION
        constexpr FloatType  pow (FloatType arg1)
        {return gridtools::gt_pow<Exponent>::apply(arg1);}
    }
#endif
    namespace expressions{
        /**Expressions defining the interface for specifiyng a given offset for a specified dimension
           \tparam Left: argument of type Dimension<>::Index, specifying the offset in the given direction*/
        template<typename Left>
        GT_FUNCTION
        constexpr typename Left::super operator +(Left d1, int const&  offset) { return typename Left::super( offset );}

        template<typename Left>
        GT_FUNCTION
        constexpr typename Left::super operator -(Left d1, int const&  offset) {return typename Left::super(-offset);}

/**@}*/
    }//namespace expressions


#ifdef CXX11_ENABLED
    namespace evaluation{

        /**\section binding_expressions (Expressions Bindings)
           @brief these functions get called by the operator () in gridtools::iterate_domain, i.e. in the functor Do method defined at the application level
           They evalueate the operator passed as argument, by recursively evaluating its arguments
           @{
        */

        /** plus evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto static constexpr value(IterateDomain const& it_domain
                                    , expr_plus<ArgType1, ArgType2> const& arg)
            -> decltype(it_domain(arg.first_operand) + it_domain(arg.second_operand)) {
            return it_domain(arg.first_operand) + it_domain(arg.second_operand);}

        /** minus evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto static constexpr value(IterateDomain const& it_domain
                                    , expr_minus<ArgType1, ArgType2> const& arg)
            -> decltype(it_domain(arg.first_operand) - it_domain(arg.second_operand)) {
            return it_domain(arg.first_operand) - it_domain(arg.second_operand);}

        /** multiplication evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto static constexpr value(IterateDomain const& it_domain
                                    , expr_times<ArgType1, ArgType2> const& arg)
            -> decltype(it_domain(arg.first_operand) * it_domain(arg.second_operand)) {
            return it_domain(arg.first_operand) * it_domain(arg.second_operand);}

        /** division evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto static constexpr value(IterateDomain const& it_domain
                                    , expr_divide<ArgType1, ArgType2> const& arg)
            -> decltype(it_domain(arg.first_operand) / it_domain(arg.second_operand)) {
            return it_domain(arg.first_operand) / it_domain(arg.second_operand);}


        /** larger of evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        bool static constexpr value(IterateDomain const& it_domain
                                    , expr_larger<ArgType1, ArgType2> const& arg){
            return it_domain(arg.first_operand) > it_domain(arg.second_operand);}

        /** larger or equal of evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        bool static constexpr value(IterateDomain const& it_domain
                                    , expr_larger_equal<ArgType1, ArgType2> const& arg){
            return it_domain(arg.first_operand) >= it_domain(arg.second_operand);}

        /** smaller of evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        bool static constexpr value(IterateDomain const& it_domain
                                    , expr_smaller<ArgType1, ArgType2> const& arg){
            return it_domain(arg.first_operand) < it_domain(arg.second_operand);}

        /** smaller or equal of evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        bool static constexpr value(IterateDomain const& it_domain
                                    , expr_smaller_equal<ArgType1, ArgType2> const& arg){
            return it_domain(arg.first_operand) <= it_domain(arg.second_operand);}

        /** and evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        bool static constexpr value(IterateDomain const& it_domain
                                    , expr_and<ArgType1, ArgType2> const& arg){
            return it_domain(arg.first_operand) && it_domain(arg.second_operand);}

        /** or evaluation*/
        template <typename IterateDomain, typename ArgType1, typename ArgType2>
        GT_FUNCTION
        bool static constexpr value(IterateDomain const& it_domain
                                    , expr_or<ArgType1, ArgType2> const& arg){
            return it_domain(arg.first_operand) || it_domain(arg.second_operand);}

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

        /**\subsection specialization (Partial Specializations)
           partial specializations for double (or float)
           @{*/
        /** sum with scalar evaluation*/
        template <typename IterateDomain, typename ArgType1, typename FloatType
                  , typename boost::enable_if<
                        typename boost::is_floating_point<FloatType>::type
                        , int >::type=0
                  >
        GT_FUNCTION
        auto static constexpr value_scalar(IterateDomain const& it_domain
                                           , expr_plus<ArgType1, FloatType> const& arg)
            -> decltype(it_domain(arg.first_operand) + arg.second_operand) {
            return it_domain(arg.first_operand) + arg.second_operand;}

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

        /** multiply with scalar evaluation*/
        template <typename IterateDomain, typename ArgType1, typename FloatType
                  , typename boost::enable_if<
                        typename boost::is_floating_point<FloatType>::type
                        , int >::type=0
                  >
        GT_FUNCTION
        auto static constexpr value_scalar(IterateDomain const& it_domain
                                           , expr_times<ArgType1, FloatType> const& arg)
            -> decltype(it_domain(arg.first_operand) * arg.second_operand) {
            return it_domain(arg.first_operand) * arg.second_operand;}

        /** divide with scalar evaluation*/
        template <typename IterateDomain
                  , typename ArgType1
                  , typename FloatType
                  , typename boost::enable_if<
                        typename boost::is_floating_point<FloatType>::type
                        , int >::type=0
                  >
        GT_FUNCTION
        auto static constexpr value_scalar(IterateDomain const& it_domain
                                           , expr_divide<ArgType1, FloatType> const& arg)
            -> decltype(it_domain(arg.first_operand) / arg.second_operand) {
            return it_domain(arg.first_operand) / arg.second_operand;}

        /** power of scalar evaluation*/
        template < typename IterateDomain
                   , typename FloatType
                   , typename IntType
                   , typename boost::enable_if<
                         typename boost::is_floating_point<FloatType>::type
                         , int >::type=0
                   , typename boost::enable_if<
                         typename boost::is_integral<IntType>::type
                         , int >::type=0
                   >
        GT_FUNCTION
        static auto constexpr value_scalar(IterateDomain const& /*it_domain*/
                                           , expr_exp<FloatType, IntType> const& arg)
            -> decltype(std::pow (arg.first_operand,  arg.second_operand)) {
            return gt_pow<2>::apply(arg.first_operand);
}


        /**
           @}
           \subsection specialization2 (Partial Specializations)
           @brief partial specializations for integer
           Here we do not use the typedef int_t, because otherwise the interface would be polluted with casting
           (the user would have to cast all the literal types (-1, 0, 1, 2 .... ) to int_t before using them in the expression)
           @{*/

        template <typename IterateDomain, typename ArgType1, typename IntType
                  , typename boost::enable_if<
                        typename boost::is_integral<IntType>::type, int >::type=0
                  >
        GT_FUNCTION
        auto static constexpr value_int(IterateDomain const& it_domain
                                        , expr_exp<ArgType1, IntType> const& arg)
            -> decltype(gt_pow<2>::apply(it_domain(arg.first_operand))) {
            return gt_pow<2>::apply(it_domain(arg.first_operand));
        }

        template <typename IterateDomain, typename ArgType1 /*typename IntType, IntType*/
                  , int exponent/*, typename boost::enable_if<typename boost::is_integral<IntType>::type, int >::type=0 */>
        GT_FUNCTION
        auto static constexpr value_int(IterateDomain const& it_domain
                                        , expr_pow<ArgType1, exponent> const& arg)
            -> decltype(gt_pow<exponent>::apply(it_domain(arg.first_operand))) {
            return gt_pow<exponent>::apply(it_domain(arg.first_operand));}

        /**@}@}*/

    }//namespace evaluation
#endif
}//namespace gridtools
