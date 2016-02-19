#pragma once

#define _META_ARRAY_VECTOR_(z,n,nil) \
    template<typename Vector, typename First,  BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType)> \
    struct meta_array_vector ## BOOST_PP_INC(n) {                       \
        typedef typename meta_array_vector ## n <typename boost::mpl::push_back<Vector , First>::type, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType)>::type type; \
    }; \
 \
    template<typename Vector, typename Mss1, typename Mss2, typename Cond, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename MssType)> \
    struct meta_array_vector ## BOOST_PP_INC(n) <Vector, condition<Mss1, Mss2, Cond>, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType) > { \
        typedef condition< \
            typename meta_array_vector ## BOOST_PP_INC(n) <Vector \
                                       , Mss1, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)>::type \
            , typename meta_array_vector ## BOOST_PP_INC(n) <Vector \
                                         , Mss2, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), MssType)>::type \
            , Cond \
            > type; \
 \
    };

namespace gridtools {

    template <typename Mss1, typename Mss2, typename Cond>
    struct condition;

    template<typename Vector>
    struct meta_array_vector0{
        typedef Vector type;
    };

    template<typename Vector, typename Mss1>
    struct meta_array_vector1{
        typedef typename boost::mpl::push_back<Vector, Mss1>::type type;
    };

    template<typename Vector, typename Mss1, typename Mss2, typename Cond>
    struct meta_array_vector1<Vector, condition<Mss1, Mss2, Cond> >{
        typedef condition<typename meta_array_vector1<Vector, Mss1>::type
                          , typename meta_array_vector1<Vector, Mss2>::type
                          , Cond>type;
    };

    // BOOST_PP_REPEAT(GT_MAX_MSS, _META_ARRAY_VECTOR_, _)

    template<typename Vector, typename First,  typename MssType0>
    struct meta_array_vector2 {
        typedef typename meta_array_vector1 <typename boost::mpl::push_back<Vector , First>::type, MssType0>::type type;
    };

    template<typename Vector, typename Mss1, typename Mss2, typename Cond,  typename MssType0>
    struct meta_array_vector2 <Vector, condition<Mss1, Mss2, Cond>, MssType0 > {
        typedef condition<
            typename meta_array_vector2<Vector
                                       , Mss1, MssType0>::type
            , typename meta_array_vector2<Vector
                                         , Mss2, MssType0>::type
            , Cond
            > type;
    };


    template<typename Vector, typename First,  typename MssType0,  typename MssType1>
    struct meta_array_vector3 {
        typedef typename meta_array_vector2 <typename boost::mpl::push_back<Vector , First>::type, MssType0, MssType1>::type type;
    };

    template<typename Vector, typename Mss1, typename Mss2, typename Cond,  typename MssType0,  typename MssType1>
    struct meta_array_vector3 <Vector, condition<Mss1, Mss2, Cond>, MssType0, MssType1 > {
        typedef condition<
            typename meta_array_vector3<Vector
                                       , Mss1, MssType0, MssType1>::type
            , typename meta_array_vector3<Vector
                                         , Mss2, MssType0, MssType1>::type
            , Cond
            > type;
    };


    template<typename Vector, typename First,  typename MssType0,  typename MssType1,  typename MssType2>
    struct meta_array_vector4 {
        typedef typename meta_array_vector3 <typename boost::mpl::push_back<Vector , First>::type, MssType0, MssType1, MssType2>::type type;
    };

    template<typename Vector, typename Mss1, typename Mss2, typename Cond,  typename MssType0,  typename MssType1,  typename MssType2>
    struct meta_array_vector4 <Vector, condition<Mss1, Mss2, Cond>, MssType0, MssType1, MssType2 > {
        typedef condition<
            typename meta_array_vector4<Vector
                                       , Mss1, MssType0, MssType1, MssType2>::type
            , typename meta_array_vector4<Vector
                                         , Mss2, MssType0, MssType1, MssType2>::type
            , Cond
            > type;
    };


    template<typename Vector, typename First,  typename MssType0,  typename MssType1,  typename MssType2,  typename MssType3>
    struct meta_array_vector5 {
        typedef typename meta_array_vector4 <typename boost::mpl::push_back<Vector , First>::type, MssType0, MssType1, MssType2, MssType3>::type type;
    };

    template<typename Vector, typename Mss1, typename Mss2, typename Cond,  typename MssType0,  typename MssType1,  typename MssType2,  typename MssType3>
    struct meta_array_vector5 <Vector, condition<Mss1, Mss2, Cond>, MssType0, MssType1, MssType2, MssType3 > {
        typedef condition<
            typename meta_array_vector5<Vector
                                       , Mss1, MssType0, MssType1, MssType2, MssType3>::type
            , typename meta_array_vector5<Vector
                                         , Mss2, MssType0, MssType1, MssType2, MssType3>::type
            , Cond
            > type;
    };

}//namespace gridtools
