#pragma once
namespace gridtools{
    //create a lazy range
    template<typename Start, typename End>
    struct lazy_range{
        typedef typename boost::mpl::push_front
        < typename lazy_range<typename boost::mpl::plus
                              <static_int<Start::value>, static_int<1> >::type, static_int<End::value> >::type
          , static_int<Start::value>
          >::type type;
    };

    //create a lazy reverse range
    template<typename Start, typename End>
    struct lazy_reverse_range{
        typedef typename boost::mpl::push_back
        < typename lazy_reverse_range<static_int<Start::value>, typename boost::mpl::minus
                                <static_int<End::value>, static_int<1> >::type >::type
          , static_int<End::value> >::type type;
    };

    //specialization anchor
    template< typename T1, typename T2, T2 t, template<typename Type, T1 val> class Start1, template<typename Type, T2 val> class Start2 >
    struct lazy_range<Start1<T1, t>, Start2<T2, t> >{
        typedef typename boost::mpl::vector1< Start1<T1, t> >::type type;
    };


    //specialization anchor
    template< typename T1, typename T2, T2 t, template<typename Type, T1 val> class Start1, template<typename Type, T2 val> class Start2 >
    struct lazy_reverse_range<Start1<T1, t>, Start2<T2, t> >{
        typedef typename boost::mpl::vector1< Start1<T1, t> >::type type;
    };
}
