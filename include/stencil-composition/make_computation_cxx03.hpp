namespace gridtools {

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


#define _MAKE_COMPUTATION(z, n, nil)                                            \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Grid                                                         \
    >                                                                           \
    computation* make_computation(                                              \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Grid& grid                                    \
    ) {                                                                         \
        return new intermediate<                                                \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Grid,POSITIONAL_WHEN_DEBUGGING                            \
        >(boost::ref(domain), grid);                                          \
    }

#else

#define _MAKE_COMPUTATION(z, n, nil)                                            \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Grid                                                         \
    >                                                                           \
    boost::shared_ptr<                                                          \
        intermediate<                                                           \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Grid ,POSITIONAL_WHEN_DEBUGGING                           \
        >                                                                       \
    > make_computation(                                                         \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Grid& grid                                    \
    ) {                                                                         \
        return boost::make_shared<                                              \
            intermediate<                                                       \
                Backend,                                                        \
                typename _impl::get_mss_array<                                  \
                BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
                >::type,                                                        \
                Domain, Grid, POSITIONAL_WHEN_DEBUGGING                       \
            >                                                                   \
        >(boost::ref(domain), grid);                                          \
    }

#endif // __CUDACC__

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_COMPUTATION, _)
#undef _MAKE_COMPUTATION


    /////////////////////////////////////////////////////////////////////////////////////
    /// MAKE POSITIONAL COMPUTATIOS
    ////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

#define _MAKE_POSITIONAL_COMPUTATION(z, n, nil)                                 \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Grid                                                         \
    >                                                                           \
    computation* make_positional_computation(                                   \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \
        Domain& domain, const Grid& grid                                    \
    ) {                                                                         \
        return new intermediate<                                                \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Grid,true                                                 \
        >(boost::ref(domain), grid);                                          \
    }

#else

#define _MAKE_POSITIONAL_COMPUTATION(z, n, nil)                                 \
    template <                                                                  \
        typename Backend,                                                       \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType),                \
        typename Domain,                                                        \
        typename Grid                                                         \
    >                                                                            \
    boost::shared_ptr<                                                          \
        intermediate<                                                           \
            Backend,                                                            \
            typename _impl::get_mss_array<                                      \
            BOOST_PP_CAT(boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
            >::type,                                                            \
            Domain, Grid ,true                                                \
        >                                                                       \
    > make_positional_computation(                                              \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType),                         \


        Domain& domain, const Grid& grid                             \
    ) {                                                                         \
        return boost::make_shared<                                              \
            intermediate<                                                       \
                Backend,                                                        \
                typename _impl::get_mss_array<                                  \
                BOOST_PP_CAT( boost::mpl::vector, BOOST_PP_INC(n)) <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)> \
                >::type,                                                        \
                Domain, Grid, true                                            \
            >                                                                   \
        >(boost::ref(domain), grid);                                          \
    }

#endif // __CUDACC__

    BOOST_PP_REPEAT(GT_MAX_MSS, _MAKE_POSITIONAL_COMPUTATION, _)
#undef _MAKE_POSITIONAL_COMPUTATION

} //namespace gridtools
