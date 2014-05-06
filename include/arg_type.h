#pragma once

#include "storage.h"
#include "layout_map.h"
#include "range.h"
#include <boost/type_traits/integral_constant.hpp>

namespace gridtools {
    /**
     * Flag to indicate that value_type will be known later
     */
    struct no_type_yet {};

    /**
     * Flag to indicate that storage will be instantiated later
     */
    struct no_storage_yet {
        typedef no_type_yet value_type;
    };

    /**
     * Flag type to identify data fields that must be treated as temporary
     */
    template <typename StorageType>
    struct temporary {
        typedef StorageType storage_type;
        typedef typename storage_type::value_type value_type;
    };

    template <>
    struct temporary<no_storage_yet> {
        typedef no_storage_yet storage_type;
        typedef storage_type::value_type value_type;
    };


    template <typename T>
    struct is_temporary {
        typedef boost::false_type type;
    };

    template <typename U>
    struct is_temporary<temporary<U> > {
        typedef boost::true_type type;
    };

    /**
     * Type to create placeholders for data fields.
     *
     * There is a specialization for the case in which T is a temporary
     *
     * @tparam I Integer index (unique) of the data field to identify it
     * @tparam T The type of the storage used to store data
     */
    template <int I, typename T>
    struct arg {
        typedef T storage_type;
        typedef typename T::iterator_type iterator_type;
        typedef typename T::value_type value_type;
        typedef boost::mpl::int_<I> index_type;
        typedef boost::mpl::int_<I> index;

        static void info() {
            std::cout << "Arg on real storage with index " << I;
        }
    };

    /**
     * Type to create placeholders for data fields.
     *
     * Specialization for the case in which T is a temporary
     *
     * @tparam I Integer index (unique) of the data field to identify it
     * @tparam T The type of values to store in the actual storage for the temporary data-field
     */
    template <int I, typename U>
    struct arg<I, temporary<U> > {
#ifndef NDEBUG
        template <typename STORAGE, typename TAG>
        struct add_tag {};

        template <typename VAL, typename LAYOUT, bool BOOL, typename OLD_TAG, template <typename, typename, bool, typename> class STORE, typename NEW_TAG>
        struct add_tag<STORE<VAL, LAYOUT, BOOL, OLD_TAG>, NEW_TAG> {
            typedef STORE<VAL, LAYOUT, true, NEW_TAG> type;
        };
#else
        template <typename STORAGE>
        struct make_it_temporary {};

        template <typename VAL, typename LAYOUT, bool BOOL, template <typename, typename, bool> class STORE>
        struct make_it_temporary<STORE<VAL, LAYOUT, BOOL> > {
            typedef STORE<VAL, LAYOUT, true> type;
        };

#endif

#ifndef NDEBUG
        typedef typename add_tag<U, arg<I, temporary<U> > >::type storage_type;
#else
        typedef typename make_it_temporary<U>::type storage_type;
#endif
        typedef typename storage_type::value_type value_type;

        typedef typename storage_type::iterator_type iterator_type;
        typedef boost::mpl::int_<I> index_type;

        static void info() {
            std::cout << "Arg on TEMP storage with index " << I;
        }
    };

    /**
     * Type to be used in elementary stencil functions to specify argument mapping and ranges
     *
     * The class also provides the interface for accessing data in the function body
     *
     * @tparam I Index of the argument in the function argument list
     * @tparam Range Bounds over which the function access the argument
     */
    template <int I, typename Range=range<0,0,0,0> >
    struct arg_type {

        template <int Im, int Ip, int Jm, int Jp, int Kp, int Km>
        struct halo {
            typedef arg_type<I> type;
        };

        int offset[3];
        typedef boost::mpl::int_<I> index_type;
        typedef Range range_type;

        GT_FUNCTION
        arg_type(int i, int j, int k) {
            offset[0] = i;
            offset[1] = j;
            offset[2] = k;
        }

        GT_FUNCTION
        arg_type() {
            offset[0] = 0;
            offset[1] = 0;
            offset[2] = 0;
        }

        GT_FUNCTION
        int i() const {
            return offset[0];
        }

        GT_FUNCTION
        int j() const {
            return offset[1];
        }

        GT_FUNCTION
        int k() const {
            return offset[2];
        }

        GT_FUNCTION
        static arg_type<I> center() {
            return arg_type<I>();
        }

        GT_FUNCTION
        const int* offset_ptr() const {
            return offset;
        }

        GT_FUNCTION
        arg_type<I> plus(int _i, int _j, int _k) const {
            return arg_type<I>(i()+_i, j()+_j, k()+_k);
        }

        static void info() {
            std::cout << "Arg_type storage with index " << I << " and range " << Range() << " ";
        }

    };

    /**
     * Struct to test if an argument is a temporary
     */
    template <typename T>
    struct is_plchldr_to_temp : boost::false_type
    {};

    /**
     * Struct to test if an argument is a temporary - Specialization yielding true
     */
    template <int I, typename T>
    struct is_plchldr_to_temp<arg<I, temporary<T> > > : boost::true_type
    {};

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for arg_type
     * @return ostream
     */
    template <int I, typename R>
    std::ostream& operator<<(std::ostream& s, arg_type<I,R> const&) {
        return s << "[ arg_type< " << I
                 << ", " << R() << " > ]";
    }

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for arg to a temporary
     * @return ostream
     */
    template <int I, typename R>
    std::ostream& operator<<(std::ostream& s, arg<I,temporary<R> > const&) {
        return s << "[ arg< " << I
                 << ", temporary<something>" << " > ]";
    }

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for arg to a NON temp
     * @return ostream
     */
    template <int I, typename R>
    std::ostream& operator<<(std::ostream& s, arg<I,R> const&) {
        return s << "[ arg< " << I
                 << ", NON TEMP" << " > ]";
    }

} // namespace gridtools
