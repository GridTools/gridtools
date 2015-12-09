namespace gridtools{

    /**@brief Expression retrieving the maximum over a specific dimension*/
    template <typename ArgType1>
    struct expr_derivative : public unary_expr<ArgType1>{
        typedef unary_expr<ArgType1> super;
        GT_FUNCTION
        constexpr expr_derivative(ArgType1 const& first_operand):super(first_operand){}

        template<typename Arg1>
        GT_FUNCTION
        constexpr expr_derivative(expr_derivative<Arg1> const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_derivative(){};
#ifndef __CUDACC__
        static char constexpr op[]=" D";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<operation, tokens::open_par, ArgType1, tokens::closed_par>;
#endif
    };

    template < typename ArgType1>
    struct is_unary_expr<expr_derivative<ArgType1> > : boost::mpl::true_ {
    };

    namespace expressions{

        template <typename ArgType1,
                  typename boost::disable_if<
                      boost::mpl::not_< boost::mpl::or_
                                        <is_accessor<ArgType1>
                                         , is_expr<ArgType1> > >
                      , int >::type=0                   >
        GT_FUNCTION
        constexpr expr_derivative<ArgType1>
        D (ArgType1 arg1){
            return expr_derivative<ArgType1>(arg1);}


    namespace evaluation{

        /** derivative evaluation*/
        template <typename IterateDomain, typename ArgType1>
        GT_FUNCTION
        static float_type
#ifdef NDEBUG
        constexpr
#endif
        value(IterateDomain const& it_domain
              , expr_derivative<ArgType1> const& arg
              , typename boost::enable_if<
              typename is_accessor<ArgType1>::type
              , int >::type=0
            )
        {
#ifndef NDEBUG
            // if(!is_accessor<ArgType1>::value)
            // {
            //     printf("derivative not supported for operator: ");
            //     ArgType1::to_string::apply();
            //     printf("\n");
            //     exit (-1);
            // }
#endif
            return 1.;
            //each operator implements its own derivative
            }

    }//namespace evaluation
    }//namespace expressions
} //namespace gridtools
