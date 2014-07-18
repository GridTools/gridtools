#ifndef _LAYOUT_MAP_H_
#define _LAYOUT_MAP_H_

#include <boost/static_assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include <assert.h>

namespace GCL {

    namespace _impl {
        template <unsigned int I>
        struct select_s;

        template <>
        struct select_s<0> {
            template <typename T>
            T& get(T & a) {
                return a;
            }

            template <typename T>
            T& get(T & a, T & /*b*/) {
                return a;
            }

            template <typename T>
            T& get(T & a, T & /*b*/, T & /*c*/) {
                return a;
            }

            template <typename T>
            T& get(T & a, T & /*b*/, T & /*c*/, T & /*d*/) {
                return a;
            }
        };

        template <>
        struct select_s<1> {
            template <typename T>
            T& get(T & /*a*/, T & b) {
                return b;
            }

            template <typename T>
            T& get(T & /*a*/, T & b, T & /*c*/) {
                return b;
            }

            template <typename T>
            T& get(T & /*a*/, T & b, T & /*c*/, T & /*d*/) {
                return b;
            }
        };

        template <>
        struct select_s<2> {
            template <typename T>
            T& get(T & /*a*/, T & /*b*/, T & c) {
                return c;
            }

            template <typename T>
            T& get(T & /*a*/, T & /*b*/, T & c, T & /*d*/) {
                return c;
            }
        };

        template <>
        struct select_s<3> {
            template <typename T>
            T& get(T & /*a*/, T & /*b*/, T & /*c*/, T & d) {
                return d;
            }
        };
    }

    template <int, int=-1, int=-1, int=-1>
        struct layout_map;

    template <int I1>
    struct layout_map<I1, -1, -1, -1> {
        static const unsigned int length=1;
        typedef typename boost::mpl::vector<boost::mpl::int_<I1> > t;

        template <unsigned int I>
            static int at() {
            BOOST_STATIC_ASSERT( I<length );
            return boost::mpl::at_c<t, I >::type::value;
        }

        int operator[](int i) {
            assert( i<length );
            switch (i) {
            case 0:
                return boost::mpl::at_c<t, 0 >::type::value;
            }
            return -1;
        }

        template <unsigned int I, typename T>
            static T select(T & a, T & b) {
            return _impl::select_s<boost::mpl::at_c<t, I >::type::value>().get(a,b);
        }

        template <unsigned int I, typename T>
            static T& find(T & a) {
            return a;
        }

    };

    template <int I1, int I2>
    struct layout_map<I1, I2, -1, -1> {
        static const unsigned int length=2;
        typedef typename boost::mpl::vector<boost::mpl::int_<I1>, 
                                            boost::mpl::int_<I2> > t;

        template <unsigned int I>
            static int at() {
            BOOST_STATIC_ASSERT( I<length );
            return boost::mpl::at_c<t, I >::type::value;
        }

        int operator[](int i) {
            assert( i<length );
            switch (i) {
            case 0:
                return boost::mpl::at_c<t, 0 >::type::value;
            case 1:
                return boost::mpl::at_c<t, 1 >::type::value;
            }
            return -1;
        }

        template <unsigned int I, typename T>
            static T& select(T & a, T & b) {
            return _impl::select_s<boost::mpl::at_c<t, I >::type::value>().get(a,b);
        }

        template <unsigned int I, typename T>
            static T& find(T & a, T & b) {
            if (boost::mpl::at_c<t, 0 >::type::value == I) {
                return a;
            } else {
                if (boost::mpl::at_c<t, 1 >::type::value == I) {
                    return b;
                }
            }
        }
    };

    /**
       Layout maps are simple sequences of integers specified
       staically. The specification happens as

       \code
       GCL::layout_map<a,b,c>
       \endcode

       where a, b, and c are integer static constants. To access the
       elements of this sequences the user should call the static method

       \code
       ::at<I>()
       \endcode

       For instance:
       \code
       GCL::layout_map<3,4,1,5>::at<2> == 1
       GCL::layout_map<3,4,1,5>::at<0> == 3
       etc.
       \endcode
    */
    template <int I1, int I2, int I3>
    struct layout_map<I1, I2, I3, -1> {
        static const unsigned int length=3;
        typedef typename boost::mpl::vector<boost::mpl::int_<I1>, 
                                            boost::mpl::int_<I2>, 
                                            boost::mpl::int_<I3> > t;
    
        template <unsigned int I>
            struct at_ {
            static const int value = boost::mpl::at_c<t, I >::type::value;
        };


        /** This function returns the value in the map that is stored at
            position 'I', where 'I' is passed in input as template
            argument.

            \tparam I The index to be queried
        */
        template <unsigned int I>
            static int at() {
            BOOST_STATIC_ASSERT( I<length );
            return boost::mpl::at_c<t, I >::type::value;
        }

        int operator[](int i) {
            assert( i<length );
            switch (i) {
            case 0:
                return boost::mpl::at_c<t, 0 >::type::value;
            case 1:
                return boost::mpl::at_c<t, 1 >::type::value;
            case 2:
                return boost::mpl::at_c<t, 2 >::type::value;
            }
            return -1;
        }

        /** Given a tuple of values and a static index, the function
            returns the reference to the value in the position indicated
            at position 'I' in the map.

            \code
            GCL::layout_map<1,2,0>::select<1>(a,b,c) == c
            \endcode

            \tparam I Index to be queried
            \param[in] a Reference to the first value
            \param[in] b Reference to the second value
            \param[in] c Reference to the third value
        */
        template <unsigned int I, typename T>
            static T& select(T & a, T & b, T & c) {
            return _impl::select_s<boost::mpl::at_c<t, I >::type::value>().get(a,b,c);
        }

        /** Given a tuple of values and a static index I, the function
            returns the reference to the element whose position
            corresponds to the position of 'I' in the map.

            \code
            GCL::layout_map<2,0,1>::find<1>(a,b,c) == c
            \endcode

            \tparam I Index to be searched in the map
            \param[in] a Reference to the first value
            \param[in] b Reference to the second value
            \param[in] c Reference to the third value
        */
        template <unsigned int I, typename T>
            static T& find(T & a, T & b, T & c) {
            if (boost::mpl::at_c<t, 0 >::type::value == I) {
                return a;
            } else {
                if (boost::mpl::at_c<t, 1 >::type::value == I) {
                    return b;
                } else {
                    if (boost::mpl::at_c<t, 2 >::type::value == I) {
                        return c;
                    }
                }
            }
        }

        /** Given a tuple of values and a static index I, the function
            returns the reference to the element whose position
            corresponds to the position of 'I' in the map.
        
            \code
            a[0] = a; a[1] = b; a[3] = c;
            GCL::layout_map<2,0,1>::find<1>(a) == c
            \endcode

            \tparam I Index to be searched in the map
            \param[in] a Pointer to a region with the elements to match
        */
        template <unsigned int I, typename T>
            static T& find(T* a) {
            return find<I>(a[0], a[1], a[2]);
        }

    };

    template <int I1, int I2, int I3, int I4>
    struct layout_map {
        static const unsigned int length=4;
        typedef typename boost::mpl::vector<boost::mpl::int_<I1>, 
                                            boost::mpl::int_<I2>, 
                                            boost::mpl::int_<I3>, 
                                            boost::mpl::int_<I4> > t;

        template <unsigned int I>
        static int at() {
            BOOST_STATIC_ASSERT( I<length );
            return boost::mpl::at_c<t, I >::type::value;
        }

        int operator[](int i) {
            assert( i<length );
            switch (i) {
            case 0:
                return boost::mpl::at_c<t, 0 >::type::value;
            case 1:
                return boost::mpl::at_c<t, 1 >::type::value;
            case 2:
                return boost::mpl::at_c<t, 2 >::type::value;
            case 3:
                return boost::mpl::at_c<t, 3 >::type::value;
            }
            return -1;
        }

        template <unsigned int I, typename T>
        static T& select(T & a, T & b, T & c, T & d) {
            return _impl::select_s<boost::mpl::at_c<t, I >::type::value>().get(a,b,c,d);
        }

        template <unsigned int I, typename T>
        static T& find(T & a, T & b, T & c, T & d) {
            if (boost::mpl::at_c<t, 0 >::type::value == I) {
                return a;
            } else {
                if (boost::mpl::at_c<t, 1 >::type::value == I) {
                    return b;
                } else {
                    if (boost::mpl::at_c<t, 2 >::type::value == I) {
                        return c;
                    } else {
                        if (boost::mpl::at_c<t, 3 >::type::value == I) {
                            return c;
                        }
                    }
                }
            }
        }

    };


    template <typename DATALO, typename PROCLO>
    struct layout_transform;

    template <int I1, int I2, int P1, int P2>
    struct layout_transform<layout_map<I1,I2>, layout_map<P1,P2> > {
        typedef layout_map<I1,I2> L1;
        typedef layout_map<P1,P2> L2;

        static const int N1 = boost::mpl::at_c<typename L1::t, P1>::type::value;
        static const int N2 = boost::mpl::at_c<typename L1::t, P2>::type::value;

        typedef layout_map<N1,N2> type;

    };

    template <int I1, int I2, int I3, int P1, int P2, int P3>
    struct layout_transform<layout_map<I1,I2,I3>, layout_map<P1,P2,P3> > {
        typedef layout_map<I1,I2,I3> L1;
        typedef layout_map<P1,P2,P3> L2;

        static const int N1 = boost::mpl::at_c<typename L1::t, P1>::type::value;
        static const int N2 = boost::mpl::at_c<typename L1::t, P2>::type::value;
        static const int N3 = boost::mpl::at_c<typename L1::t, P3>::type::value;

        typedef layout_map<N1,N2,N3> type;

    };

    template <int D>
    struct default_layout_map;

    template <>
    struct default_layout_map<1> {
        typedef layout_map<0> type;
    };

    template <>
    struct default_layout_map<2> {
        typedef layout_map<0,1> type;
    };

    template <>
    struct default_layout_map<3> {
        typedef layout_map<0,1,2> type;
    };

    template <>
    struct default_layout_map<4> {
        typedef layout_map<0,1,2,3> type;
    };
}



#endif

