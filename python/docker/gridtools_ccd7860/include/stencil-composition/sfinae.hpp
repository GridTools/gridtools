#pragma once

namespace gridtools {

    /**
       @brief Sobstitution Failure is Not An Error

       design pattern used to detect at compile-time whether a class contains a member or not (introspection)
    */
    // define an SFINAE structure
    template <typename T>
    struct SFINAE;

    template <>
    struct SFINAE<int>{};

#ifdef CXX11_ENABLED
#define HAS_TYPE_SFINAE( name, has_name, get_name )                     \
    template<typename TFunctor>                                         \
    struct has_name                                                     \
    {                                                                   \
        struct MixIn                                                    \
        {                                                               \
            using name = int ;                                          \
        };                                                              \
        struct derived : public TFunctor, public MixIn {};              \
                                                                        \
                                                                        \
        template<typename TDerived>                                     \
            static boost::mpl::false_ test( SFINAE<typename TDerived::name>* x ); \
        template<typename TDerived>                                     \
            static boost::mpl::true_ test(...);                         \
                                                                        \
            typedef decltype(test<derived>(0)) type;                    \
            typedef TFunctor functor_t;                                 \
    };                                                                  \
                                                                        \
    template<typename Functor>                                          \
    struct get_name{                                                    \
        typedef typename Functor::name type;                            \
    };
#else
#define HAS_TYPE_SFINAE( name, has_name, get_name )                     \
    template<typename TFunctor>                                         \
    struct has_name                                                     \
    {                                                                   \
        struct MixIn                                                    \
        {                                                               \
            typedef int name ;                                          \
        };                                                              \
        struct derived : public TFunctor, public MixIn {};              \
                                                                        \
                                                                        \
        template<typename TDerived>                                     \
            static boost::mpl::false_ test( SFINAE<typename TDerived::name>* x ); \
        template<typename TDerived>                                     \
            static boost::mpl::true_ test(...);                         \
                                                                        \
        typedef BOOST_TYPEOF(test<derived>(0)) type;                    \
        typedef TFunctor functor_t;                                     \
    };                                                                  \
                                                                        \
    template<typename Functor>                                          \
    struct get_name{                                                    \
        typedef typename Functor::name type;                            \
    };
#endif


    /** SFINAE method to check if a class has a method named "name" which is constexpr and returns an int*/
#define HAS_STATIC_METHOD_SFINAE( name )                               \
    template<int> struct sfinae_true : std::true_type{};        \
    template<class T>                                           \
    sfinae_true<(T::name(), 0)> test(int);                        \
    template<class>                                             \
    std::false_type test(...);                                  \
                                                                \
    template<class T>                                           \
    struct has_constexpr_name : decltype(detail::check<T>(0)){};


    /**@brief Implementation of introspection

     returning true when the template functor has a type alias called 'xrange'.
     This type defines a range used in order to arbitrarily extend/shrink the loop bounds
     for the current functor at compile-time.
     NOTE: it does not work yet for the blocked strategy. This because in that case it is not trivial
     to modify the loop bounds with 'functor' granularity. Further thinking-refactoring needed for that case
    */
    HAS_TYPE_SFINAE(xrange, has_xrange, get_xrange)

    /**@brief Implementation of introspection

     returning true when the template functor has a type alias called 'xrange'.
     This type defines a range used in order to arbitrarily extend/shrink the loop bounds
     for the current functor at compile-time.
     NOTE: it does not work yet for the blocked strategy. This because in that case it is not trivial
     to modify the loop bounds with 'functor' granularity. Further thinking-refactoring needed for that case
    */
    HAS_TYPE_SFINAE(xrange_subdomain, has_xrange_subdomain, get_xrange_subdomain)


    /*use with eval_if as follows*/
    // typedef typename boost::mpl::eval_if_c<has_xrange<functor_type>::type::value
    //                                        , get_xrange< functor_type >
    //                                        , boost::mpl::identity<range<0,0,0> > >::type new_range_t;

    // typedef typename boost::mpl::eval_if_c<has_xrange_subdomain<functor_type>::type::value
    //                                        , get_xrange_subdomain< functor_type >
    //                                        , boost::mpl::identity<range<0,0,0> > >::type xrange_subdomain_t;

}
