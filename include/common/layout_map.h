#ifndef _LAYOUT_MAP_H_
#define _LAYOUT_MAP_H_

#include <boost/static_assert.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include "../common/gt_assert.h"
#include "../common/host_device.h"
#include "../common/defs.h"

/**
@file
@brief definifion of the data layout
Here are defined the classes select_s and layout_map.
 */
namespace gridtools {

    namespace _impl {
        template <uint_t I>
        struct select_s;

        template <>
        struct select_s<0> {
            template <typename T>
            GT_FUNCTION
            T& get(T & a) {
                return a;
            }

            template <typename T>
            GT_FUNCTION
            T& get(T & a, T & /*b*/) {
                return a;
            }

            template <typename T>
            GT_FUNCTION
            T& get(T & a, T & /*b*/, T & /*c*/) {
                return a;
            }

            template <typename T>
            GT_FUNCTION
            T& get(T & a, T & /*b*/, T & /*c*/, T & /*d*/) {
                return a;
            }
        };

        template <>
        struct select_s<1> {
            template <typename T>
            GT_FUNCTION
            T& get(T & /*a*/, T & b) {
                return b;
            }

            template <typename T>
            GT_FUNCTION
            T& get(T & /*a*/, T & b, T & /*c*/) {
                return b;
            }

            template <typename T>
            GT_FUNCTION
            T& get(T & /*a*/, T & b, T & /*c*/, T & /*d*/) {
                return b;
            }
        };

        template <>
        struct select_s<2> {
            template <typename T>
            GT_FUNCTION
            T& get(T & /*a*/, T & /*b*/, T & c) {
                return c;
            }

            template <typename T>
            GT_FUNCTION
            T& get(T & /*a*/, T & /*b*/, T & c, T & /*d*/) {
                return c;
            }
        };

        template <>
        struct select_s<3> {
            template <typename T>
            GT_FUNCTION
            T& get(T & /*a*/, T & /*b*/, T & /*c*/, T & d) {
                return d;
            }
        };
    }

/**
@struct
@brief Used as template argument in the storage.
In particular in the \ref gridtools::base_storage class it regulate memory access order, defined at compile-time, by leaving the interface unchanged.
*/
    template <short_t, short_t=-1, short_t=-1, short_t=-1>
        struct layout_map;

    template <short_t I1>
    struct layout_map<I1, -1, -1, -1> {
        static const ushort_t length=1;
        typedef boost::mpl::vector1_c<short_t, I1> t;

        template <ushort_t I>
        GT_FUNCTION
        static short_t at() {
            BOOST_STATIC_ASSERT( I<length );
            return boost::mpl::at_c<t, I >::type::value;
        }

        GT_FUNCTION
        short_t operator[](short_t i) {
            assert( i<length );
            switch (i) {
            case 0:
                return boost::mpl::at_c<t, 0 >::type::value;
            }
            return -1;
        }

        template <ushort_t I, typename T>
        GT_FUNCTION
        static T select(T & a, T & b) {
            return _impl::select_s<boost::mpl::at_c<t, I >::type::value>().get(a,b);
        }

        template <ushort_t I, typename T>
        GT_FUNCTION
            static uint_t& find(uint_t & a) {
            return a;
        }

    };

    template <short_t I1, short_t I2>
    struct layout_map<I1, I2, -1, -1> {
        static const ushort_t length=2;
        typedef boost::mpl::vector2_c<short_t, I1, I2> t;

        template <ushort_t I>
        GT_FUNCTION
        static short_t at() {
            BOOST_STATIC_ASSERT( I<length );
            return boost::mpl::at_c<t, I >::type::value;
        }

        GT_FUNCTION
            short_t operator[](short_t i) {
            assert( i<length );
            switch (i) {
            case 0:
                return boost::mpl::at_c<t, 0 >::type::value;
            case 1:
                return boost::mpl::at_c<t, 1 >::type::value;
            }
            return -1;
        }

        template <ushort_t I>
        GT_FUNCTION
            static uint_t& select(uint_t & a, uint_t & b) {
            return _impl::select_s<boost::mpl::at_c<t, I >::type::value>().get(a,b);
        }

        template <ushort_t I>
        GT_FUNCTION
        static uint_t& find(uint_t & a, uint_t & b) {
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
       statically. The specification happens as

       \code
       gridtools::layout_map<a,b,c>
       \endcode

       where a, b, and c are integer static constants. To access the
       elements of this sequences the user should call the static method

       \code
       ::at<I>()
       \endcode

       For instance:
       \code
       gridtools::layout_map<3,4,1,5>::at<2> == 1
       gridtools::layout_map<3,4,1,5>::at<0> == 3
       etc.
       \endcode
    */
    template <short_t I1, short_t I2, short_t I3>
    struct layout_map<I1, I2, I3, -1> {
        static  const ushort_t length=3;
        typedef boost::mpl::vector3_c<short_t, I1, I2, I3> t;

      template <ushort_t i>
        GT_FUNCTION
	static constexpr short_t get() {
	  return boost::mpl::at_c<t, i >::type::value;
	}


        template <ushort_t I>
        struct at_ {
            static const short_t value = boost::mpl::at_c<t, I >::type::value;
        };

        // Gives the position at which I is.
        template <ushort_t I>
        struct pos_ {

            template <short_t X, bool IsHere>
            struct _find_pos
            {
                static const short_t value = _find_pos<X+1, boost::mpl::at_c<t, X+1 >::type::value == I>::value;
            };

            template <short_t X>
            struct _find_pos<X, true> {
                static const short_t value = X;
            };

            template <bool IsHere>
            struct _find_pos<3, IsHere> {
                static const short_t value = -1;
            };

            static const short_t value = _find_pos<0, boost::mpl::at_c<t, 0 >::type::value == I>::value;

        };

        /** This function returns the value in the map that is stored at
            position 'I', where 'I' is passed in input as template
            argument.

            \tparam I The index to be queried
        */
        template <ushort_t I>
        GT_FUNCTION
        static short_t at() {
            BOOST_STATIC_ASSERT( I<length );
            return boost::mpl::at_c<t, I >::type::value;
        }

        GT_FUNCTION
        short_t operator[](short_t i) {
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
            gridtools::layout_map<1,2,0>::select<1>(a,b,c) == c
            \endcode

            \tparam I Index to be queried
            \param[in] a Reference to the first value
            \param[in] b Reference to the second value
            \param[in] c Reference to the third value
        */
        template <ushort_t I, typename T>
        GT_FUNCTION
        static T& select(T & a, T & b, T & c) {
            return _impl::select_s<boost::mpl::at_c<t, I >::type::value>().get(a,b,c);
        }

        /** Given a tuple of values and a static index I, the function
            returns the reference to the element whose position
            corresponds to the position of 'I' in the map.

            \code
            gridtools::layout_map<2,0,1>::find<1>(a,b,c) == c
            \endcode

            \tparam I Index to be searched in the map
            \param[in] a Reference to the first value
            \param[in] b Reference to the second value
            \param[in] c Reference to the third value
        */
        template <ushort_t I>
        GT_FUNCTION
            static uint_t& find(uint_t & a, uint_t & b, uint_t & c) {
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
            assert(true);
            return a; // killing warnings by nvcc
        }

        /** Given a tuple of values and a static index I, the function
            returns the reference to the element whose position
            corresponds to the position of 'I' in the map.

            This version works with const&

            \code
            GCL::layout_map<2,0,1>::find<1>(a,b,c) == c
            \endcode

            \tparam I Index to be searched in the map
            \param[in] a Reference to the first value
            \param[in] b Reference to the second value
            \param[in] c Reference to the third value
        */
        template <ushort_t I>
        GT_FUNCTION
        static uint_t const& find(uint_t const& a, uint_t const& b, uint_t const& c) {
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
            return a; // killing warnings by nvcc
        }

        /** Given a tuple of values and a static index I, the function
            returns the reference to the element whose position
            corresponds to the position of 'I' in the map.

            \code
            a[0] = a; a[1] = b; a[3] = c;
            gridtools::layout_map<2,0,1>::find<1>(a) == c
            \endcode

            \tparam I Index to be searched in the map
            \param[in] a Pointer to a region with the elements to match
        */
        template <ushort_t I>
        GT_FUNCTION
        static const uint_t& find (const uint_t* a)  {
            return find<I>(a[0], a[1], a[2]);
        }

    };

    template <short_t I1, short_t I2, short_t I3, short_t I4>
    struct layout_map {
        static const ushort_t length=4;
        typedef boost::mpl::vector4_c<short_t, I1, I2, I3, I4> t;

        template <ushort_t I>
        GT_FUNCTION
        static int at() {
            BOOST_STATIC_ASSERT( I<length );
            return boost::mpl::at_c<t, I >::type::value;
        }

        GT_FUNCTION
        int operator[](short_t i) {
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

      template <short_t i>
      static constexpr ushort_t get() {
	return boost::mpl::at_c<t, i >::type::value;
      }

        template <ushort_t I, typename T>
        GT_FUNCTION
        static T& select(T & a, T & b, T & c, T & d) {
            return _impl::select_s<boost::mpl::at_c<t, I >::type::value>().get(a,b,c,d);
        }

        template <ushort_t I, typename T>
        GT_FUNCTION
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
            return -1; // killing warnings by nvcc
        }

    };


    template <typename DATALO, typename PROCLO>
    struct layout_transform;

    template <short_t I1, short_t I2, short_t P1, short_t P2>
    struct layout_transform<layout_map<I1,I2>, layout_map<P1,P2> > {
        typedef layout_map<I1,I2> L1;
        typedef layout_map<P1,P2> L2;

        static const short_t N1 = boost::mpl::at_c<typename L1::t, P1>::type::value;
        static const short_t N2 = boost::mpl::at_c<typename L1::t, P2>::type::value;

        typedef layout_map<N1,N2> type;

    };

    template <short_t I1, short_t I2, short_t I3, short_t P1, short_t P2, short_t P3>
    struct layout_transform<layout_map<I1,I2,I3>, layout_map<P1,P2,P3> > {
        typedef layout_map<I1,I2,I3> L1;
        typedef layout_map<P1,P2,P3> L2;

        static const short_t N1 = boost::mpl::at_c<typename L1::t, P1>::type::value;
        static const short_t N2 = boost::mpl::at_c<typename L1::t, P2>::type::value;
        static const short_t N3 = boost::mpl::at_c<typename L1::t, P3>::type::value;

        typedef layout_map<N1,N2,N3> type;

    };

    template <short_t D>
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
} // namespace gridtools



#endif
