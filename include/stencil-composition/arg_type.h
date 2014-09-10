#pragma once

#include "../storage/storage.h"
#include "../storage/host_tmp_storage.h"
#include "../common/layout_map.h"
#include "range.h"
#include <boost/type_traits/integral_constant.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <vector>
#include "../common/is_temporary_storage.h"

namespace gridtools {

    /**
     * Type to indicate that the type is not decided yet
     */
    template <typename RegularStorageType>
    struct no_storage_type_yet {
        typedef void storage_type;
        typedef typename RegularStorageType::iterator_type iterator_type;
        typedef typename RegularStorageType::value_type value_type;
        static void text() {
            std::cout << "text: no_storage_type_yet<" << RegularStorageType() << ">" << std::endl;
        }

        //std::string name() {return std::string("no_storage_yet NAMEname");}

        void info() const {
            std::cout << "No sorage type yet for storage type " << RegularStorageType() << std::endl;
        }
    };

    template <typename RST>
    std::ostream& operator<<(std::ostream& s, no_storage_type_yet<RST>) {
        return s << "no_storage_type_yet<" << RST() << ">" ;
    }

    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>  > : public boost::true_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};



    template <enumtype::backend X, typename T, typename U>
    struct is_storage<base_storage<X,T,U,true>  *  > : public boost::false_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};



    template <enumtype::backend X, typename T, typename U>
    struct is_storage<base_storage<X,T,U,false>  *  > : public boost::true_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

    template <typename U>
    struct is_storage<no_storage_type_yet<U>  *  > : public boost::false_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>* > : public boost::true_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>& > : public boost::true_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

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

    namespace enumtype
    {
        namespace{
        template <int Coordinate>
            struct T{
            T(int val){ value=val;}
            static const int direction=Coordinate;
            int value;
        };
        }
        
	typedef T<0> x;
	typedef T<1> y;
	typedef T<2> z;
    }


    GT_FUNCTION
    struct initialize
    {
        initialize(int* offset) : m_offset(offset)
            {}

        template<typename X>
        inline void operator( )(X i) const {
            m_offset[X::direction] = i.value;
        }
        int* m_offset;
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
    struct arg_type   {

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

#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
#warning "Obsolete version of the GCC compiler"
      // GCC compiler bug solved in versions 4.9+, Clang is OK, the others were not tested
      // while waiting for an update in nvcc (which is not supporting gcc 4.9 at present)
      // we implement a suboptimal solution
      template <typename X1, typename X2, typename X3 >
        GT_FUNCTION
	  arg_type ( X1 x, X2 y, X3 z){
	boost::fusion::vector<X1, X2, X3> vec(x, y, z);
            boost::fusion::for_each(vec, initialize(offset));
        }

      template <typename X1, typename X2 >
        GT_FUNCTION
	  arg_type ( X1 x, X2 y){
	boost::fusion::vector<X1, X2> vec(x, y);
            boost::fusion::for_each(vec, initialize(offset));
        }

      template <typename X1>
        GT_FUNCTION
	  arg_type ( X1 x){
	boost::fusion::vector<X1> vec(x);
            boost::fusion::for_each(vec, initialize(offset));
        }

#else
      //if you get a compiler error here, use the version above
        template <typename... X >
        GT_FUNCTION
        arg_type ( X... x){
	  boost::fusion::vector<X...> vec(x...);
	  boost::fusion::for_each(vec, initialize(offset));
        }
#endif

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
    struct is_plchldr_to_temp; //: boost::false_type

    /**
     * Struct to test if an argument is a temporary no_storage_type_yet - Specialization yielding true
     */
    template <int I, typename T>
    struct is_plchldr_to_temp<arg<I, no_storage_type_yet<T> > > : boost::true_type
    {};


    template <int I, enumtype::backend X, typename T, typename U>
    struct is_plchldr_to_temp<arg<I, base_storage<X, T, U,  true> > > : boost::true_type
    {};

    template <int I, enumtype::backend X, typename T, typename U>
    struct is_plchldr_to_temp<arg<I, base_storage< X, T, U,false> > > : boost::false_type
    {};

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for arg_type
     * @return ostream
     */
    template <int I, typename R>
    std::ostream& operator<<(std::ostream& s, arg_type<I,R> const& x) {
        return s << "[ arg_type< " << I
                 << ", " << R()
                 << " (" << x.i()
                 << ", " << x.j()
                 << ", " << x.k()
                 <<" ) > ]";
    }

    template <int I, typename R>
    std::ostream& operator<<(std::ostream& s, arg<I,no_storage_type_yet<R> > const&) {
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
