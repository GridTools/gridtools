/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include <gridtools.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include "../common/gt_assert.hpp"
#include "../common/host_device.hpp"
#include "../common/defs.hpp"
#include "../common/array.hpp"

/**
   @file
   @brief definifion of the data layout
   Here are defined the classes select_s and layout_map.
*/
namespace gridtools {

    /**
       @struct
       @brief Used as template argument in the storage.
       In particular in the \ref gridtools::base_storage class it regulate memory access order, defined at compile-time,
       by leaving the interface unchanged.
    */

    namespace _impl {
        template < ushort_t I >
        struct select_s;

        template <>
        struct select_s< 0 > {
            template < typename T >
            GT_FUNCTION T &get(T &a) {
                return a;
            }

            template < typename T >
            GT_FUNCTION T &get(T &a, T & /*b*/) {
                return a;
            }

            template < typename T >
            GT_FUNCTION T &get(T &a, T & /*b*/, T & /*c*/) {
                return a;
            }

            template < typename T >
            GT_FUNCTION T &get(T &a, T & /*b*/, T & /*c*/, T & /*d*/) {
                return a;
            }
        };

        template <>
        struct select_s< 1 > {
            template < typename T >
            GT_FUNCTION T &get(T & /*a*/, T &b) {
                return b;
            }

            template < typename T >
            GT_FUNCTION T &get(T & /*a*/, T &b, T & /*c*/) {
                return b;
            }

            template < typename T >
            GT_FUNCTION T &get(T & /*a*/, T &b, T & /*c*/, T & /*d*/) {
                return b;
            }
        };

        template <>
        struct select_s< 2 > {
            template < typename T >
            GT_FUNCTION T &get(T & /*a*/, T & /*b*/, T &c) {
                return c;
            }

            template < typename T >
            GT_FUNCTION T &get(T & /*a*/, T & /*b*/, T &c, T & /*d*/) {
                return c;
            }
        };

        template <>
        struct select_s< 3 > {
            template < typename T >
            GT_FUNCTION T &get(T & /*a*/, T & /*b*/, T & /*c*/, T &d) {
                return d;
            }
        };
    }

    /**
       @struct
       @brief Used as template argument in the storage.
       In particular in the \ref gridtools::base_storage class it regulate memory access order, defined at compile-time,
       by leaving the interface unchanged.
    */
    template < short_t, short_t = -2, short_t = -2, short_t = -2 >
    struct layout_map;

    template < typename layout >
    struct is_layout_map : boost::mpl::false_ {};
    template < short_t t1, short_t t2, short_t t3, short_t t4 >
    struct is_layout_map< layout_map< t1, t2, t3, t4 > > : boost::mpl::true_ {};

    template < short_t I1 >
    struct layout_map< I1, -2, -2, -2 > {
        static const ushort_t length = 1;
        typedef boost::mpl::vector4_c< short_t, I1, -2, -2, -2 > layout_vector_t;

        template < ushort_t I >
        GT_FUNCTION static short_t at() {
            BOOST_STATIC_ASSERT(I < length);
            return boost::mpl::at_c< layout_vector_t, I >::type::value;
        }

        GT_FUNCTION
        short_t operator[](short_t i) {
            assert(i < length);
            switch (i) {
            case 0:
                return boost::mpl::at_c< layout_vector_t, 0 >::type::value;
            }
            return -1;
        }

        template < ushort_t I, typename T >
        GT_FUNCTION static T select(T &a, T &b) {
            return _impl::select_s< boost::mpl::at_c< layout_vector_t, I >::type::value >().get(a, b);
        }

        template < ushort_t I, typename T >
        GT_FUNCTION static T &find(T &a) {
            return a;
        }
    };

    template < short_t I1, short_t I2 >
    struct layout_map< I1, I2, -2, -2 > {
        static const ushort_t length = 2;
        typedef boost::mpl::vector4_c< short_t, I1, I2, -2, -2 > layout_vector_t;

        template < ushort_t I >
        GT_FUNCTION static short_t at() {
            BOOST_STATIC_ASSERT(I < length);
            return boost::mpl::at_c< layout_vector_t, I >::type::value;
        }

        GT_FUNCTION
        short_t operator[](short_t i) {
            assert(i < length);
            switch (i) {
            case 0:
                return boost::mpl::at_c< layout_vector_t, 0 >::type::value;
            case 1:
                return boost::mpl::at_c< layout_vector_t, 1 >::type::value;
            }
            return -1;
        }

        template < ushort_t I, typename T >
        GT_FUNCTION static T &select(T &a, T &b) {
            return _impl::select_s< boost::mpl::at_c< layout_vector_t, I >::type::value >().get(a, b);
        }

        template < ushort_t I, typename T >
        GT_FUNCTION static T const &find(T const &a, T const &b) {
            if (boost::mpl::at_c< layout_vector_t, 0 >::type::value == I) {
                return a;
            } else {
                if (boost::mpl::at_c< layout_vector_t, 1 >::type::value == I) {
                    return b;
                }
            }
        }

        template < ushort_t I, typename T >
        GT_FUNCTION static uint_t find(const T *indices) {
            BOOST_STATIC_ASSERT(I < length);
            return find< I, T >(indices[0], indices[1]);
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
    template < short_t I1, short_t I2, short_t I3 >
    struct layout_map< I1, I2, I3, -2 > {
        static const short_t length = 3;
        typedef boost::mpl::vector4_c< short_t, I1, I2, I3, -2 > layout_vector_t;

        template < short_t I >
        struct at_ {
            static const short_t value = boost::mpl::at_c< layout_vector_t, I >::type::value;
        };

        template < short_t I, short_t DefaultVal >
        struct at_default {
            static const short_t _value = boost::mpl::at_c< layout_vector_t, I >::type::value;
            static const short_t value = (_value < 0) ? DefaultVal : _value;
        };

        // Gives the position at which I is. e.g., I want to know which is the stride of i (0)?
        // then if pos_<0> is 0, then the index i has stride 1, and so on ...
        template < short_t I >
        struct pos_ {

            template < short_t X, bool IsHere >
            struct _find_pos {
                static const short_t value =
                    _find_pos< X + 1, boost::mpl::at_c< layout_vector_t, X + 1 >::type::value == I >::value;
            };

            template < short_t X >
            struct _find_pos< X, true > {
                static const short_t value = X;
            };

            template < bool IsThere >
            struct _find_pos< length + 1, IsThere > {
                static const short_t value = -2; // value_is_not_there___print_a_compiler_error value = X;
            };

            template < bool IsHere >
            struct _find_pos< length, IsHere > {
                static const short_t value = -1;
            };

            static const short_t value =
                _find_pos< 0, boost::mpl::at_c< layout_vector_t, 0 >::type::value == I >::value;
        };

        /** This function returns the value in the map that is stored at
            position 'I', where 'I' is passed in input as template
            argument.

            \tparam I The index to be queried
        */
        template < short_t I >
        GT_FUNCTION static short_t at() {
            BOOST_STATIC_ASSERT(I < length);
            return boost::mpl::at_c< layout_vector_t, I >::type::value;
        }

        GT_FUNCTION
        short_t operator[](short_t i) {
            assert(i < length);
            switch (i) {
            case 0:
                return boost::mpl::at_c< layout_vector_t, 0 >::type::value;
            case 1:
                return boost::mpl::at_c< layout_vector_t, 1 >::type::value;
            case 2:
                return boost::mpl::at_c< layout_vector_t, 2 >::type::value;
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
        template < short_t I, typename T >
        GT_FUNCTION static T &select(T &a, T &b, T &c) {
            return _impl::select_s< boost::mpl::at_c< layout_vector_t, I >::type::value >().get(a, b, c);
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
        template < short_t I, typename T >
        GT_FUNCTION static T &find(T &a, T &b, T &c) {
            if (boost::mpl::at_c< layout_vector_t, 0 >::type::value == I) {
                return a;
            } else {
                if (boost::mpl::at_c< layout_vector_t, 1 >::type::value == I) {
                    return b;
                } else {
                    if (boost::mpl::at_c< layout_vector_t, 2 >::type::value == I) {
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
        template < short_t I, typename T >
        GT_FUNCTION static T const &find(T const &a, T const &b, T const &c) {
            if (boost::mpl::at_c< layout_vector_t, 0 >::type::value == I) {
                return a;
            } else {
                if (boost::mpl::at_c< layout_vector_t, 1 >::type::value == I) {
                    return b;
                } else {
                    if (boost::mpl::at_c< layout_vector_t, 2 >::type::value == I) {
                        return c;
                    }
                }
            }
            assert(false);
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
        template < short_t I, typename T >
        GT_FUNCTION static T &find(T *a) {
            return find< I >(a[0], a[1], a[2]);
        }

        /** Given a tuple of values and a static index I, the function
            returns the value of the element whose position
            corresponds to the position of 'I' in the map. If the
            value is not found a default value is returned, which is
            passed as template parameter. It works for intergal types.

            Default value is picked by default if C++11 is anabled,
            otherwise it has to be provided.

            \code
            gridtools::layout_map<2,0,1>::find_val<1,type,default>(a,b,c) == c
            \endcode

            \tparam I Index to be searched in the map
            \tparam Default_Val Default value returned if the find is not successful
            \param[in] a Reference to the first value
            \param[in] b Reference to the second value
            \param[in] c Reference to the third value
        */
        template < ushort_t I, typename T, T DefaultVal >
        GT_FUNCTION static T find_val(T const &a, T const &b, T const &c) {
            if ((uint_t)boost::mpl::at_c< layout_vector_t, 0 >::type::value == I) {
                return a;
            } else {
                if ((uint_t)boost::mpl::at_c< layout_vector_t, 1 >::type::value == I) {
                    return b;
                } else {
                    if ((uint_t)boost::mpl::at_c< layout_vector_t, 2 >::type::value == I) {
                        return c;
                    }
                }
            }

            return DefaultVal;
        }

        /** Given a tuple of values and a static index I, the function
            returns the value of the element whose position
            corresponds to the position of 'I' in the map.

            Default value is picked by default if C++11 is anabled,
            otherwise it has to be provided.

            \code
            a[0] = a; a[1] = b; a[3] = c;
            gridtools::layout_map<2,0,1>::find<1>(a) == c
            \endcode

            \tparam I Index to be searched in the map
            \param[in] a Pointer to a region with the elements to match
        */
        template < ushort_t I, typename T, T DefaultVal >
        GT_FUNCTION static T find_val(T const *a) {
            return find_val< I, T, DefaultVal >(a[0], a[1], a[2]);
        }

        /** Given a tuple of values and a static index I, the function
            returns the value of the element whose position
            corresponds to the position of 'I' in the map. If the
            value is not found a default value is returned, which is
            passed as template parameter. It works for intergal types.

            Default value is picked by default if C++11 is anabled,
            otherwise it has to be provided.

            \code
            gridtools::layout_map<2,0,1>::find_val<1,type,default>(a,b,c) == c
            \endcode

            \tparam I Index to be searched in the map
            \tparam Default_Val Default value returned if the find is not successful
            \tparam[in] Indices List of argument where to return the found value
            \param[in] indices List of values (length must be equal to the length of the layout_map length)
        */
        template < ushort_t I, typename T, T DefaultVal, typename Tuple >
        GT_FUNCTION static T find_val(Tuple const &indices) {
            if ((pos_< I >::value >= length)) {
                return DefaultVal;
            } else {
                assert((int_t)Tuple::n_dim - (int_t)pos_< I >::value - 1 >= 0);
                // GRIDTOOLS_STATIC_ASSERT((Tuple::n_dim-pos_<I>::value-1) >= 0, "accessing a tuple of offsets with a
                // negative index");
                // GRIDTOOLS_STATIC_ASSERT((Tuple::n_dim-pos_<I>::value-1) < Tuple::n_dim, "accessing a tuple of offsets
                // out of bounds");
                return indices.template get< Tuple::n_dim - pos_< I >::value - 1 >();
            }
        }
    };

    template < short_t I1, short_t I2, short_t I3, short_t I4 >
    struct layout_map {
        static const short_t length = 4;
        typedef boost::mpl::vector4_c< short_t, I1, I2, I3, I4 > layout_vector_t;

        template < short_t I >
        GT_FUNCTION static short_t at() {
            BOOST_STATIC_ASSERT(I < length);
            return boost::mpl::at_c< layout_vector_t, I >::type::value;
        }

        GT_FUNCTION
        short_t operator[](short_t i) {
            assert(i < length);
            switch (i) {
            case 0:
                return boost::mpl::at_c< layout_vector_t, 0 >::type::value;
            case 1:
                return boost::mpl::at_c< layout_vector_t, 1 >::type::value;
            case 2:
                return boost::mpl::at_c< layout_vector_t, 2 >::type::value;
            case 3:
                return boost::mpl::at_c< layout_vector_t, 3 >::type::value;
            }
            return -1;
        }

        template < short_t I, typename T >
        GT_FUNCTION static T &select(T &a, T &b, T &c, T &d) {
            return _impl::select_s< boost::mpl::at_c< layout_vector_t, I >::type::value >().get(a, b, c, d);
        }

        template < short_t I, typename T >
        GT_FUNCTION static T &find(T &a, T &b, T &c, T &d) {
            if (boost::mpl::at_c< layout_vector_t, 0 >::type::value == I) {
                return a;
            } else {
                if (boost::mpl::at_c< layout_vector_t, 1 >::type::value == I) {
                    return b;
                } else {
                    if (boost::mpl::at_c< layout_vector_t, 2 >::type::value == I) {
                        return c;
                    } else {
                        if (boost::mpl::at_c< layout_vector_t, 3 >::type::value == I) {
                            return c;
                        }
                    }
                }
            }
            return -1; // killing warnings by nvcc
        }
    };
} // namespace gridtools
