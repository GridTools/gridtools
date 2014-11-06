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

    //Decorator is the GPU storage
    template <typename BaseType , template <typename T> class Decorator >
    struct is_storage<Decorator<BaseType>  *  > : public is_storage<typename BaseType::basic_type*>
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};


    //Decorator is the integrator
    template <typename First, typename ... BaseType , template <typename ... T> class Decorator >
      struct is_storage<Decorator<First, BaseType...>  *  > : public is_storage<typename First::basic_type*>
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

    //Decorator is the integrator
    template <typename BaseType , template <typename T, ushort_t O> class Decorator, ushort_t Order >
    struct is_storage<Decorator<BaseType, Order>  *  > : public is_storage<typename BaseType::basic_type*>
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
    template <uint_t I, typename T>
    struct arg {
        typedef T storage_type;
        typedef typename T::iterator_type iterator_type;
        typedef typename T::value_type value_type;
        typedef static_uint<I> index_type;
        typedef static_uint<I> index;

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
        namespace impl{
        template <ushort_t Coordinate>
            struct T{
            T(short_t const& val){ value=val;}
            static const ushort_t direction=Coordinate;
            short_t value;
        };
        }

        typedef impl::T<0> x;
        typedef impl::T<1> y;
        typedef impl::T<2> z;
    }

    struct initialize
    {
        GT_FUNCTION
        initialize(int_t* offset) : m_offset(offset)
            {}

        template<typename X>
        GT_FUNCTION
        inline void operator( )(X const& i) const {
            m_offset[X::direction] = i.value;
        }
        int_t* m_offset;
    };

    /**
     * Type to be used in elementary stencil functions to specify argument mapping and ranges
     *
     * The class also provides the interface for accessing data in the function body
     *
     * @tparam I Index of the argument in the function argument list
     * @tparam Range Bounds over which the function access the argument
     */
  template <uint_t I, typename Range=range<0,0,0,0>, ushort_t dimension=3 >
    struct arg_type   {

        template <uint_t Im, uint_t Ip, uint_t Jm, uint_t Jp, uint_t Kp, uint_t Km>
        struct halo {
            typedef arg_type<I> type;
        };

    int_t offset[dimension]
#ifdef CXX11_ENABLED
    ={0}
#endif
      ;

        typedef static_uint<I> index_type;
        typedef Range range_type;

        GT_FUNCTION
	arg_type(int_t i, int_t j, int_t k) {
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
#ifndef CXX11_ENABLED
	offset[0]=0;
	offset[1]=0;
	offset[2]=0;
#endif
          boost::fusion::vector<X1, X2> vec(x, y);
          boost::fusion::for_each(vec, initialize(offset));
      }

      template <typename X1>
        GT_FUNCTION
	  arg_type ( X1 x){
#ifndef CXX11_ENABLED
	offset[0]=0;
	offset[1]=0;
	offset[2]=0;
#endif
	boost::fusion::vector<X1> vec(x);
	boost::fusion::for_each(vec, initialize(offset));
        }

#else
      //#ifdef CXX11_ENABLED
      //if you get a compiler error here, use the version above
        template <typename... X >
        GT_FUNCTION
        arg_type ( X... x){
            //BOOST_STATIC_ASSERT(sizeof...(X)<=dimensions);
            boost::fusion::vector<X...> vec(x...);
            boost::fusion::for_each(vec, initialize(offset));
        }

        // template <typename... int_t >
        // GT_FUNCTION
        // arg_type ( int_t... x){
	//   BOOST_STATIC_ASSERT(sizeof...(X)==dimensions);
	//   offset={x};
        // }
      //#endif //CXX11_ENABLED
#endif //__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)

        GT_FUNCTION
        arg_type() {
            offset[0] = 0;
            offset[1] = 0;
            offset[2] = 0;
        }

        GT_FUNCTION
        int_t i() const {
            return offset[0];
        }

        GT_FUNCTION
        int_t j() const {
            return offset[1];
        }

        GT_FUNCTION
        int_t k() const {
            return offset[2];
        }

        GT_FUNCTION
        static arg_type<I> center() {
            return arg_type<I>();
        }

        GT_FUNCTION
        const int_t* offset_ptr() const {
            return offset;
        }

        GT_FUNCTION
        arg_type<I> plus(int_t _i, int_t _j, int_t _k) const {
            return arg_type<I>(i()+_i, j()+_j, k()+_k);
        }

        static void info() {
            std::cout << "Arg_type storage with index " << I << " and range " << Range() << " ";
        }

        // Methods to stop the recursion when dealing with extra dimensions
        static const ushort_t n_args=0;

        template<ushort_t index>
            GT_FUNCTION
            int_t n() const {//stop recursion
                printf("The dimension you are trying to access exceeds the number of dimensions by %d.\n ", index+1);
                exit (-1);
            }

    };



//################################################################################
//                              Multidimensionality
//################################################################################

    namespace enumtype{
        template<ushort_t N>
        struct Extra{
	  GT_FUNCTION
	  Extra(short_t const& val){ value=val;}
	  static const ushort_t direction=N;
	  short_t value;
         };
    }

    /**@brief this is a decorator of the arg_type, which is matching the extra dimensions
       \param n_args is the current ID of the extra dimension
       \param index_type is the index of the storage type
     */
    template< class ArgType>
    struct arg_decorator : public ArgType{

        typedef ArgType super;
        static const ushort_t n_args=super::n_args+1;
        typedef typename super::index_type index_type;

#ifdef CXX11_ENABLED
	/**@brief constructor taking an integer as the first argument, and then other optional arguments.
	   The integer gets assigned to the current extra dimension and the other arguments are passed to the base class (in order to get assigned to the other dimensions). When this constructor is used all the arguments have to be specified and passed to the function call in order. No check is done on the order*/
        template <typename... Whatever>
        GT_FUNCTION
        arg_decorator ( int_t const& t, Whatever... x): super( x... ) {
	  m_offset=t;
	}

	/**@brief constructor taking the Extra class as argument.
	 This allows to specify the extra arguments out of order. Note that 'enumtype::Extra' is a
	 language keyword used at the interface level.*/
        template <ushort_t Idx, typename... Whatever>
        GT_FUNCTION
        arg_decorator ( enumtype::Extra<Idx> const& t, Whatever... x): super( x... ) {

	  //if the following check is not true, you specified an extra index exceeding the dimenison of the field
	  BOOST_STATIC_ASSERT(enumtype::Extra<Idx>::direction<=n_args);

            //there's no need to allow further flexibility on memory layout (?), i.e. extra dimensions memory location will be undefined
	  if(enumtype::Extra<Idx>::direction==n_args)
            {
                // printf("offset %d was specified to be %d \n", n_args, t.value);
	      m_offset=t.value;
            }
            else
            {
                // printf("no offset was specified for extra dimension %d \n", t.direction);
                m_offset=0;
            }
        }

	/**@brief fallback constructor, when the others are not taken, meaning that no offset for the extra dimension was specified, simply forwards the call to the constructor of the base class.*/
        template <typename... Whatever>
        GT_FUNCTION
        arg_decorator ( Whatever... x ): super( x... ) {
	  BOOST_STATIC_ASSERT(sizeof...(x)<=n_args);
            // printf("no offsets for extra dimension was specified (but there are %d) \n", n_args);
            m_offset=0;
        }//just forward

#else //CXX11_ENABLED
whatever not compiling
#endif

    /** @brief usage: n<3>() returns the offset of extra dimension 3
	loops recursively over the children, decreasing each time the index, until it has reached the dimension matching the index specified as template argument.
     */
    template<short_t index>
    GT_FUNCTION
    short_t n() const {//recursively travel the list of offsets
    BOOST_STATIC_ASSERT( index>0 );
    // printf("index to the n method:%d \n", index);
    BOOST_STATIC_ASSERT( index<=n_args );
    //this might not be compile-time efficient for large indexes,
    //because both taken and not taken branches are compiled
    return index==1? m_offset : super::template n<index-1>();
    }

    private:
      short_t m_offset;
    };


    template <typename ArgType, uint_t Number>
      struct arg_extend{
      typedef arg_decorator<typename arg_extend<ArgType, Number-1>::type>  type;
      };

    template<typename ArgType>
    struct arg_extend<ArgType, 0>{typedef ArgType type;};

//################################################################################
//                              Compile time checks
//################################################################################

    /**
     * Struct to test if an argument is a temporary
     */
    template <typename T>
    struct is_plchldr_to_temp;

    /**
     * Struct to test if an argument is a temporary no_storage_type_yet - Specialization yielding true
     */
    template <uint_t I, typename T>
    struct is_plchldr_to_temp<arg<I, no_storage_type_yet<T> > > : boost::true_type
    {};

    template <uint_t I, enumtype::backend X, typename T, typename U>
    struct is_plchldr_to_temp<arg<I, base_storage<X, T, U,  true> > > : boost::true_type
    {};

    template <uint_t I, enumtype::backend X, typename T, typename U>
    struct is_plchldr_to_temp<arg<I, base_storage< X, T, U,false> > > : boost::false_type
    {};

  //here the decorator is the GPU storage or the dimension extension
    template <uint_t I, typename BaseType, template <typename T> class Decorator>
      struct is_plchldr_to_temp<arg<I, Decorator<BaseType> > > : is_plchldr_to_temp<arg<I, typename BaseType::basic_type> >
    {};


  //here the decorator is the dimension extension
    template <uint_t I, typename First, typename ... BaseType, template <typename ... T> class Decorator>
      struct is_plchldr_to_temp<arg<I, Decorator<First, BaseType ...> > > : is_plchldr_to_temp<arg<I, typename First::basic_type> >
    {};

  /* //here the decorator is the dimension extension */
  /*   template <uint_t I, typename BaseType, template <typename T, ushort_t O> class Decorator, ushort_t Order> */
  /*     struct is_plchldr_to_temp<arg<I, Decorator<BaseType , Order> > > : is_plchldr_to_temp<arg<I, typename BaseType::basic_type> > */
  /*   {}; */

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for arg_type
     * @return ostream
     */
    template <uint_t I, typename R>
    std::ostream& operator<<(std::ostream& s, arg_type<I,R> const& x) {
        return s << "[ arg_type< " << I
                 << ", " << R()
                 << " (" << x.i()
                 << ", " << x.j()
                 << ", " << x.k()
                 <<" ) > ]";
    }

    template <uint_t I, typename R>
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
    template <uint_t I, typename R>
    std::ostream& operator<<(std::ostream& s, arg<I,R> const&) {
        return s << "[ arg< " << I
                 << ", NON TEMP" << " > ]";
    }

    /**@brief Expression templates definition.
       The expression templates are a method to parse at compile time the mathematical expression given
       by the user, recognizing the structure and building a syntax tree by recursively nesting
       templates.
     */
    template <typename ArgType1, typename ArgType2>
    struct expr{
        GT_FUNCTION
        expr(ArgType1 const& first_operand, ArgType2 const& second_operand)
            :
            first_operand(first_operand),
            second_operand(second_operand)
            {}

        ArgType1 const first_operand;
        ArgType2 const second_operand;
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_plus : public expr<ArgType1, ArgType2>{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        expr_plus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_minus : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        expr_minus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_times : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        expr_times(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_divide : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        expr_divide(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
    };

#ifdef CXX11_ENABLED
    namespace expressions{
        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        expr_plus<ArgType1, ArgType2 >  operator + (ArgType1 arg1, ArgType2 arg2){return expr_plus<ArgType1, ArgType2 >(std::forward<ArgType1>(arg1), std::forward<ArgType2>(arg2));}

        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        expr_minus<ArgType1, ArgType2 > operator - (ArgType1 arg1, ArgType2 arg2){return expr_minus<ArgType1, ArgType2 >(arg1, arg2);}

        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        expr_times<ArgType1, ArgType2 > operator * (ArgType1 arg1, ArgType2 arg2){return expr_times<ArgType1, ArgType2 >(arg1, arg2);}

        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        expr_divide<ArgType1, ArgType2 > operator / (ArgType1 arg1, ArgType2 arg2){return expr_divide<ArgType1, ArgType2 >(arg1, arg2);}

    }//namespace expressions

#endif
} // namespace gridtools
