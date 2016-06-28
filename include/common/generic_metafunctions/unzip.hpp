#pragma once

namespace gridtools {

    template < class Container1, class Container2, typename... Zipped >
    struct do_unzip;

    template < template < typename... T > class Container,
        typename First,
        typename... UnzippedFirst,
        typename Second,
        typename... UnzippedSecond,
        typename... Zipped >
    struct do_unzip< Container< UnzippedFirst... >, Container< UnzippedSecond... >, First, Second, Zipped... > {
        typedef typename do_unzip< Container< UnzippedFirst..., First >,
            Container< UnzippedSecond..., Second >,
            Zipped... >::first first;
        typedef typename do_unzip< Container< UnzippedFirst..., First >,
            Container< UnzippedSecond..., Second >,
            Zipped... >::second second;
    };

    template < template < typename... T > class Container,
        typename First,
        typename Second,
        typename... UnzippedFirst,
        typename... UnzippedSecond >
    struct do_unzip< Container< UnzippedFirst... >, Container< UnzippedSecond... >, First, Second > {
        typedef Container< UnzippedFirst..., First > first;
        typedef Container< UnzippedSecond..., Second > second;
    };

    template < typename ZippedSequence >
    struct unzip;

    /**
       @brief metafunction unzipping a generic container built with a parameter pack

       (a1, b1, a2, b2, a3, b3,...)->(a1,a2,a3,...),(b1,b2,b3,...)
     */
    template < template < typename... T > class Sequence, typename First, typename Second, typename... Args >
    struct unzip< Sequence< First, Second, Args... > > {
        typedef typename do_unzip< Sequence< First >, Sequence< Second >, Args... >::first first;
        typedef typename do_unzip< Sequence< First >, Sequence< Second >, Args... >::second second;
    };
}
