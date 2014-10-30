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

    template<typename ArgType, typename Storage>
    struct arg_storage_pair {
        typedef ArgType arg_type;
        typedef Storage storage_type;

        Storage *ptr;

        arg_storage_pair(Storage* p)
            : ptr(p)
        {}

        Storage* operator*() {
            return ptr;
        }
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

        template<typename Storage>
        arg_storage_pair<arg<I,T>, Storage>
        operator=(Storage& ref) {
            BOOST_MPL_ASSERT( (boost::is_same<Storage, T>) );
            return arg_storage_pair<arg<I,T>, Storage>(&ref);
        }

        static void info() {
            std::cout << "Arg on real storage with index " << I;
        }
    };

    namespace enumtype
    {
        namespace{
        template <int Coordinate>
            struct T{
            constexpr T(int val) : value
#if( (!defined(CXX11_ENABLED)) && (!defined(__CUDACC__ )))
	    (val)
#else
	      {val}
#endif
{}
            static const int direction=Coordinate;
            int value;
        };
        }

	typedef T<0> x;
	typedef T<1> y;
	typedef T<2> z;
    }

    template <int N, typename X>
    constexpr int initialize( X x )
    {
        return (X::direction==N? x.value : 0);
    }

#ifdef CXX11_ENABLED
    template <int N, typename X, typename ... Rest>
    constexpr int initialize(X x, Rest ... rest )
    {
        return X::direction==N? x.value : initialize<N>(rest...);
    }
#else
    template <int N, typename X, typename Y>
    constexpr int initialize(X x, Y y)
    {
        return X::direction==N? x.value : Y::direction==N? y.value : 0;
    }

    template <int N, typename X, typename Y, typename Z>
    constexpr int initialize(X x, Y y, Z z)
    {
        return X::direction==N? x.value : Y::direction==N? y.value : Z::direction==N? z.value : 0;
    }
#endif

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

#ifdef CXX11_ENABLED
        int m_offset[3]={0,0,0};
#else
        int m_offset[3];
#endif
        typedef boost::mpl::int_<I> index_type;
        typedef Range range_type;

        GT_FUNCTION
        constexpr arg_type(int i, int j, int k)
#if( (!defined(CXX11_ENABLED)) && (!defined(__CUDACC__ )))
      {
	m_offset[0]=i;
	m_offset[1]=j;
	m_offset[2]=k;
      }
#else
	  : m_offset{i,j,k} {}
#endif

        constexpr arg_type(arg_type const& other)
#if( (!defined(CXX11_ENABLED)) && (!defined(__CUDACC__ )))
      {
	m_offset[0]=other.m_offset[0];
	m_offset[1]=other.m_offset[1];
	m_offset[2]=other.m_offset[2];
      }
#else
:m_offset{other.m_offset[0], other.m_offset[1], other.m_offset[2]}{}
#endif

#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
#warning "Obsolete version of the GCC compiler"
      // GCC compiler bug solved in versions 4.9+, Clang is OK, the others were not tested
      // while waiting for an update in nvcc (which is not supporting gcc 4.9 at present)
      // we implement a suboptimal solution
      template <typename X1, typename X2, typename X3 >
        GT_FUNCTION
	  arg_type ( X1 x, X2 y, X3 z)
#if( (!defined(CXX11_ENABLED)) && (!defined(__CUDACC__ )))
      {
	m_offset[0]=initialize<0>(x,y,z);
	m_offset[1]=initialize<1>(x,y,z);
	m_offset[2]=initialize<2>(x,y,z);
      }
#else
:m_offset{initialize<0>(x,y,z), initialize<1>(x,y,z), initialize<2>(x,y,z)}{ }
#endif
      template <typename X1, typename X2 >
        GT_FUNCTION
	  constexpr arg_type ( X1 x, X2 y)
#if( (!defined(CXX11_ENABLED)) && (!defined(__CUDACC__ )))
      {
	m_offset[0]=initialize<0>(x,y);
	m_offset[1]=initialize<1>(x,y);
	m_offset[2]=initialize<2>(x,y);
      }
#else
:m_offset{initialize<0>(x,y), initialize<1>(x,y), initialize<2>(x,y)}{ }
#endif

      template <typename X1>
        GT_FUNCTION
	  constexpr arg_type ( X1 x)
#if( (!defined(CXX11_ENABLED)) && (!defined(__CUDACC__ )))
      {
	m_offset[0]=initialize<0>(x);
	m_offset[1]=initialize<1>(x);
	m_offset[2]=initialize<2>(x);
      }
#else
:m_offset{initialize<0>(x), initialize<1>(x), initialize<2>(x)}{ }
#endif

#else // __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)

//      if you get a compiler error here, use the version above
        template <typename... X >
        GT_FUNCTION
        constexpr arg_type ( X... x):m_offset{initialize<0>(x...), initialize<1>(x...), initialize<2>(x...)}{
        }
#endif //__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)

        GT_FUNCTION
        constexpr arg_type()
#if( (!defined(CXX11_ENABLED)) && (!defined(__CUDACC__ )))
      {
	m_offset[0]=0;
	m_offset[1]=0;
	m_offset[2]=0;
      }
#else
:m_offset{0,0,0} {}
#endif

        GT_FUNCTION
        constexpr int i() const {
            return m_offset[0];
        }

        GT_FUNCTION
        constexpr int j() const {
            return m_offset[1];
        }

        GT_FUNCTION
        constexpr int k() const {
            return m_offset[2];
        }

        GT_FUNCTION
        static constexpr  arg_type<I> center() {
            return arg_type<I>();
        }

        GT_FUNCTION
        constexpr int const* offset_ptr() const {
            return &m_offset[0];
        }

        GT_FUNCTION
        constexpr arg_type<I> plus(int _i, int _j, int _k) const {
            return arg_type<I>(i()+_i, j()+_j, k()+_k);
        }

        static  void info() {
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

    template <typename ArgType1, typename ArgType2>
    struct expr{
        GT_FUNCTION
        constexpr expr(ArgType1 const& first_operand, ArgType2 const& second_operand)
            :
#if( (!defined(CXX11_ENABLED)) && (!defined(__CUDACC__ )))
      first_operand(first_operand),
	second_operand(second_operand)
#else
            first_operand{first_operand},
            second_operand{second_operand}
#endif
            {}

        ArgType1 const first_operand;
        ArgType2 const second_operand;
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_plus : public expr<ArgType1, ArgType2>{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_plus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_minus : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_minus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_times : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_times(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_divide : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_divide(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
    };

#ifdef CXX11_ENABLED
    namespace expressions{
        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_plus<ArgType1, ArgType2 >  operator + (ArgType1 arg1, ArgType2 arg2){return expr_plus<ArgType1, ArgType2 >(std::forward<ArgType1>(arg1), std::forward<ArgType2>(arg2));}

        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_minus<ArgType1, ArgType2 > operator - (ArgType1 arg1, ArgType2 arg2){return expr_minus<ArgType1, ArgType2 >(arg1, arg2);}

        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_times<ArgType1, ArgType2 > operator * (ArgType1 arg1, ArgType2 arg2){return expr_times<ArgType1, ArgType2 >(arg1, arg2);}

        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_divide<ArgType1, ArgType2 > operator / (ArgType1 arg1, ArgType2 arg2){return expr_divide<ArgType1, ArgType2 >(arg1, arg2);}

    }//namespace expressions

#endif
} // namespace gridtools
