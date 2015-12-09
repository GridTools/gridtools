namespace gridtools{

    /**@brief Expression retrieving the maximum over a specific dimension*/
    template <typename ArgType1>
    struct expr_max : public unary_expr<ArgType1>{
        typedef unary_expr<ArgType1> super;
        GT_FUNCTION
        constexpr expr_max(ArgType1 const& first_operand):super(first_operand){}

        template<typename Arg1>
        GT_FUNCTION
        constexpr expr_max(expr_max<Arg1> const& other):super(other){}

#ifndef __CUDACC__
    private:
#endif
        GT_FUNCTION
        constexpr expr_max(){};
#ifndef __CUDACC__
        static char constexpr op[]=" max";
        typedef string_c<print, op> operation;
    public:
        //currying and recursion (this gets inherited)
        using to_string = concatenate<operation, tokens::open_par, ArgType1, tokens::closed_par>;
#endif
    };

    template < typename ArgType1>
    struct is_unary_expr<expr_max<ArgType1> > : boost::mpl::true_ {
    };

    namespace expression{
        GT_FUNCTION
        constexpr expr_max<ArgType1,
                           typename boost::disable_if<
                               boost::mpl::not_< boost::mpl::or_
                                                 <is_accessor<ArgType1>
                                                  , is_expr<ArgType1> > >
                               , int >::type=0 >
        max (ArgType1 arg1){
            return expr_max<ArgType1>(arg1);}

    }

    namespace evaluation{

        /** plus evaluation*/
        template <typename IterateDomain, typename ArgType1, typename Dimension>
        GT_FUNCTION
        auto static constexpr value(IterateDomain const& it_domain
                                    , expr_max<ArgType1> const& arg)
            -> decltype(it_domain(arg.first_operand) ) {
            const auto N=it_domain.get().get_storage_dim(arg.fist_operand)[Dimension::direction-1];
            // all the offsets discarded??
            auto max = it_domain(ArgType1(Dimension()));
            for(int i=0; i<N; ++i){
                max = max>it_domain(ArgType1(Dimension(i))) ? max : it_domain(ArgType1(Dimension())) ;
                }
            return max;
        }

    }//namespace evaluation
} //namespace gridtools
