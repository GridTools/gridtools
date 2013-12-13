#pragma once

#include "storage.h"
#include "layout_map.h"
#include "range.h"
#include <boost/type_traits/integral_constant.hpp>

namespace gridtools {
    /**
     * Flag type to identify data fields that must be treated as temporary
     */
    template <typename t_value_type>
    struct temporary {
        typedef t_value_type value_type;
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

        static arg<I,T> center() {
            return arg<I,T>();
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
        typedef U value_type;
        typedef storage<U, gridtools::layout_map<0,1,2>, true, arg<I, temporary<U> > > storage_type;
        typedef typename storage_type::iterator_type iterator_type;
        typedef boost::mpl::int_<I> index_type;
    };

    /**
     * Type to be used in elementary stencil functions to specify argument mapping and ranges
     * 
     * The class also provides the interface for accessing data in the function body
     * 
     * @tparam I Index of the argument in the function argument list
     * @tparam t_range Bounds over which the function access the argument
     */
    template <int I, typename t_range=range<0,0,0,0> >
    struct arg_type {

        template <int im, int ip, int jm, int jp, int kp, int km>
        struct halo {
            typedef arg_type<I> type;
        };

        int offset[3];
        typedef typename boost::mpl::int_<I> index_type;
        typedef t_range range_type;

        arg_type(int i, int j, int k) {
            offset[0] = i;
            offset[1] = j;
            offset[2] = k;
        }

        arg_type() {
            offset[0] = 0;
            offset[1] = 0;
            offset[2] = 0;
        }

        int i() const {
            return offset[0];
        }

        int j() const {
            return offset[1];
        }

        int k() const {
            return offset[2];
        }

        static arg_type<I> center() {
            return arg_type<I>();
        }

        const int* offset_ptr() const {
            return offset;
        }

        arg_type<I> plus(int _i, int _j, int _k) const {
            return arg_type<I>(i()+_i, j()+_j, k()+_k);
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
