#pragma once


namespace gridtools {

    template<typename T>
    struct if_condition_extract_index_t{
        typedef T type;
    };


    template<typename T1, typename T2, typename Cond>
    struct if_condition_extract_index_t<condition<T1, T2, Cond> > {
        typedef Cond type;
    };

    template <typename Conditional>
    struct fill_conditionals_set;

    template <>
    struct fill_conditionals_set<boost::mpl::true_>{
        /**@brief recursively assigning all the conditional in the corresponding fusion vector*/
        template<typename ConditionalsSet, typename First, typename Second, typename ... Mss>
        static void apply(ConditionalsSet& set_, First const& first_, Second second_, Mss const& ... args_){

            // if(is_condition<First>::value)
            boost::fusion::at_key<First>(set_)=conditional<First::index_t::value>(first_.value());

            /*binary subtree traversal*/
            fill_conditionals_set< typename is_condition<typename First::first_t>::type >::apply(set_, first_.first());
            fill_conditionals_set< typename is_condition<typename First::second_t>::type >::apply(set_, first_.second());

            /*go to other branch*/
            fill_conditionals_set<boost::mpl::has_key<ConditionalsSet, typename Second::index_t> >::apply(set_, second_, args_ ...);
        }

        /**recursion anchor*/
        template<typename ConditionalsSet, typename First>
        static void apply(ConditionalsSet& set_, First const& first_){

            //if(is_conditional<First>::value)
            boost::fusion::at_key<typename First::index_t>(set_) = conditional<First::index_t::index_value>(first_.value());

            /*binary subtree traversal*/
            fill_conditionals_set< typename is_condition<typename First::first_t>::type >::apply(set_, first_.first());
            fill_conditionals_set< typename is_condition<typename First::second_t>::type >::apply(set_, first_.second());

        }
    };


    template <>
    struct fill_conditionals_set<boost::mpl::false_>{
        /** recursion anchor */
        template<typename ConditionalsSet, typename First, typename ... Mss>
        static void apply(ConditionalsSet& set_, First const& first_, Mss const& ... args_){
            // fill_conditionals_set<boost::mpl::has_key<ConditionalSet, Second> >::apply(set_, second_, args_ ...);
        }

        /**recursion anchor*/
        template<typename ConditionalsSet, typename First>
        static void apply(ConditionalsSet& set_, First const& first_){
        }

    };

    namespace _impl {
        /**
         * @brief metafunction that extracts a meta array with all the mss descriptors found in the Sequence of types
         * @tparam Sequence sequence of types that contains some mss descriptors
         */
        template<typename Sequence>
        struct get_mss_array
        {
            GRIDTOOLS_STATIC_ASSERT(( boost::mpl::is_sequence<Sequence>::value ), "Internal Error: wrong type");

            typedef typename boost::mpl::fold<
                Sequence,
                boost::mpl::vector0<>,
                boost::mpl::eval_if<
                    is_mss_descriptor<boost::mpl::_2>,
                    boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>,
                    boost::mpl::_1
                    >
                >::type mss_vector;

            typedef meta_array<mss_vector, boost::mpl::quote1<is_mss_descriptor> > type;
        };
    } //namespace _impl

    template<typename Condition>
    struct condition_to_conditional{
        GRIDTOOLS_STATIC_ASSERT(is_condition<Condition>::value, "wrong type");
        typedef conditional<Condition::index_t::index_t::value> type;
    };

    /*recursion anchor*/
    template<typename ConditionalsSet>
    static void fill_conditionals(ConditionalsSet& set_){
    }

    template<typename ConditionalsSet, typename First, typename ... Mss>
    static void fill_conditionals(ConditionalsSet& set_, First const& first_, Mss const& ... args_){
        fill_conditionals_set<typename boost::mpl::has_key<ConditionalsSet, typename if_condition_extract_index_t<First>::type>::type >::apply(set_, first_, args_ ...);
        fill_conditionals(set_, args_ ...);
    }


    /**reaching a leaf, while recursively constructing the conditinals_set*/
    template<typename Vec, typename Mss>
    struct construct_conditionals_set{
        typedef Vec type;

    };

    /**recursively traverse the binary tree to construct the conditionals_set*/
    template<typename Vec, typename Mss1, typename Mss2, typename Cond>
    struct construct_conditionals_set<Vec, condition<Mss1, Mss2, Cond> >{
        typedef typename construct_conditionals_set<
            typename construct_conditionals_set<
                typename boost::mpl::push_back<Vec, Cond >::type, Mss1>::type
            , Mss2>::type type;
    };


    template <
        typename Backend,
        typename Domain,
        typename Grid,
        typename ... Mss
        >
    computation* make_computation (
        Domain& domain,
        const Grid& grid,
        Mss ... args_
        ) {

        /*TODO traverse the subtree*/
        typedef typename boost::mpl::fold< boost::mpl::vector<Mss ...>
                                           , boost::mpl::vector0<>
                                           , boost::mpl::if_<
                                               is_condition<boost::mpl::_2>
                                                 , construct_conditionals_set<
                                                     boost::mpl::_1
                                                       , boost::mpl::_2 >
                                                 , boost::mpl::_1>
                                           >::type conditionals_set_mpl_t;

        typedef typename boost::fusion::result_of::as_set<conditionals_set_mpl_t>::type conditionals_set_t;
        conditionals_set_t conditionals_set_;

        fill_conditionals(conditionals_set_, args_ ...);

        return new intermediate<
            Backend
            ,
            meta_array<typename meta_array_vector<boost::mpl::vector0<>, Mss ...>::type, boost::mpl::quote1<is_mss_descriptor> >
            //typename _impl::get_mss_array<
            // boost::mpl::vector<Mss ...>
            //  >::type
            , Domain
            , Grid
            , conditionals_set_t
            , false
            >(domain, grid, conditionals_set_);

    }
}
