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

    //Decorator is the storage
    template <typename BaseType , template <typename T> class Decorator >
    struct is_storage<Decorator<BaseType>  *  > : public is_storage<typename BaseType::basic_type*>
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

#ifdef CXX11_ENABLED
    //Decorator is the integrator
    template <typename First, typename ... BaseType , template <typename ... T> class Decorator >
      struct is_storage<Decorator<First, BaseType...>  *  > : public is_storage<typename First::basic_type*>
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};
#endif

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
      //namespace impl{
        template <ushort_t Coordinate>
            struct Dimension{

	      GT_FUNCTION
            constexpr Dimension(int_t val) : value
#if( (!defined(CXX11_ENABLED)) )
	    (val)
#else
	      {val}
#endif
{}
            static const ushort_t direction=Coordinate;
            int_t value;
        };
	//}

        typedef Dimension<0> x;
        typedef Dimension<1> y;
        typedef Dimension<2> z;
    }

    template <ushort_t N, typename X>
      GT_FUNCTION
    constexpr int_t initialize( X x )
    {
        return (X::direction==N? x.value : 0);
    }

#ifdef CXX11_ENABLED
    template <ushort_t N, typename X, typename ... Rest>
      GT_FUNCTION
    constexpr int_t initialize(X x, Rest ... rest )
    {
        return X::direction==N? x.value : initialize<N>(rest...);
    }
#else
    template <ushort_t N, typename X, typename Y>
      GT_FUNCTION
    constexpr int_t initialize(X x, Y y)
    {
        return X::direction==N? x.value : Y::direction==N? y.value : 0;
    }

    template <ushort_t N, typename X, typename Y, typename Z>
      GT_FUNCTION
      constexpr int_t initialize(X x, Y y, Z z)
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
  template <uint_t I, typename Range=range<0,0,0,0>, ushort_t dimension=3 >
    struct arg_type   {

        template <uint_t Im, uint_t Ip, uint_t Jm, uint_t Jp, uint_t Kp, uint_t Km>
        struct halo {
            typedef arg_type<I> type;
        };

    int_t m_offset[dimension]
#ifdef CXX11_ENABLED
        ={0}
#endif
      ;

        typedef static_uint<I> index_type;
        typedef Range range_type;

      /**NOTE: templating on the int type, because if we use int_t directly, and if int_t is different from int, then the user would have to explicitly specify the cast to int_t*/
      template <typename IntType>
        GT_FUNCTION
        constexpr arg_type(IntType i, IntType j, IntType k)
#if( (!defined(CXX11_ENABLED)))
      {
          m_offset[0]=i;
          m_offset[1]=j;
          m_offset[2]=k;
      }
#else
	  : m_offset{i,j,k} {}
#endif


#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
#warning "Obsolete version of the GCC compiler"
      // GCC compiler bug solved in versions 4.9+, Clang is OK, the others were not tested
      // while waiting for an update in nvcc (which is not supporting gcc 4.9 at present)
      // we implement a suboptimal solution
      template <typename X1, typename X2, typename X3 >
        GT_FUNCTION
	  arg_type ( X1 x, X2 y, X3 z)
#if( (!defined(CXX11_ENABLED)))
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
#if( (!defined(CXX11_ENABLED)))
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
#if( (!defined(CXX11_ENABLED)) )
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


      /**NOTE: the following constructor when used with the brace initializer produces with nvcc a considerable amount of extra instructions, and degrades the performances (which is probably a compiler bug, I couldn't reproduce it on a small test).*/
      GT_FUNCTION
#if( (!defined(CXX11_ENABLED)) || (defined(__CUDACC__ )))
      explicit arg_type()
      {
      	m_offset[0]=0;
      	m_offset[1]=0;
      	m_offset[2]=0;
      }
#else
      constexpr explicit arg_type()
 : m_offset{0}  {}
#endif

        GT_FUNCTION
        constexpr int_t i() const {
            return m_offset[0];
        }

        GT_FUNCTION
        constexpr int_t j() const {
            return m_offset[1];
        }

        GT_FUNCTION
        constexpr int_t k() const {
            return m_offset[2];
        }

        GT_FUNCTION
        static constexpr  arg_type<I> center() {
            return arg_type<I>();
        }

        GT_FUNCTION
        constexpr int_t const* offset_ptr() const {
            return &m_offset[0];
        }

        GT_FUNCTION
        constexpr arg_type<I> plus(int_t _i, int_t _j, int_t _k) const {
            return arg_type<I>(i()+_i, j()+_j, k()+_k);
        }

        static  void info() {
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

    /* namespace enumtype{ */
    /*     template<ushort_t N> */
    /*     struct Extra{ */
    /* 	  GT_FUNCTION */
    /* 	  Extra(int_t const& val){ value=val;} */
    /* 	  static const ushort_t direction=N; */
    /* 	  int_t value; */
    /*      }; */
    /* } */

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
        template <typename IntType, typename... Whatever>
        GT_FUNCTION
        arg_decorator ( IntType const& t, Whatever... x): super( x... ) {
	  m_offset=t;
	}

	/* //implicit cast */
        /* template <typename... Whatever> */
        /* GT_FUNCTION */
        /* arg_decorator ( int const& t, Whatever... x): super( x... ) { */
	/*   m_offset=t; */
	/* } */

	/**@brief constructor taking the Extra class as argument.
	 This allows to specify the extra arguments out of order. Note that 'enumtype::Extra' is a
	 language keyword used at the interface level.*/
        template <ushort_t Idx, typename... Whatever>
        GT_FUNCTION
        arg_decorator ( enumtype::Dimension<Idx> const& t, Whatever... x): super( x... ) {

	  //if the following check is not true, you specified an extra index exceeding the dimenison of the field
	  //BOOST_STATIC_ASSERT(enumtype::Dimension<Idx>::direction<=n_args);

            //there's no need to allow further flexibility on memory layout (?), i.e. extra dimensions memory location will be undefined
	  if(enumtype::Dimension<Idx>::direction==n_args)
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

// #else //CXX11_ENABLED
// whatever not compiling
#endif

    /** @brief usage: n<3>() returns the offset of extra dimension 3
	loops recursively over the children, decreasing each time the index, until it has reached the dimension matching the index specified as template argument.
	Note that here the offset we are talking about here looks the same as the offsets for the arg_type, but it implies actually a change of the base storage pointer.
     */
    template<short_t index>
    GT_FUNCTION
  short_t n() const {//recursively travel the list of offsets
    BOOST_STATIC_ASSERT( index>0 );
    // printf("index to the n method:%d \n", index);
    BOOST_STATIC_ASSERT( index<=n_args );
    //this might not be compile-time efficient for large indexes,
    //because both taken and not taken branches are compiled. boost::mpl::if would be better.
    return index==1? m_offset : super::template n<index-1>();
    }

/* /\** returns the offset in the storage pointers array corresponding to a specified dimension (the offset e.g. for dimension 3 is n<0>()+n<1>()+n<2>()+m_offset, because all the storage pointers lie internally on a 1D array)*\/ */
/*  template<short_t index> */
/*    GT_FUNCTION */
/*     short_t offset() const {//recursively travel the list of offsets */
/*     BOOST_STATIC_ASSERT( index>0 ); */
/*     BOOST_STATIC_ASSERT( index<=n_args ); */
/*     ????? */
/*  } */

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

  //here the decorator is the storage
    template <uint_t I, typename BaseType, template <typename T> class Decorator>
      struct is_plchldr_to_temp<arg<I, Decorator<BaseType> > > : is_plchldr_to_temp<arg<I, typename BaseType::basic_type> >
    {};

#ifdef CXX11_ENABLED
  //here the decorator is the dimension extension
    template <uint_t I, typename First, typename ... BaseType, template <typename ... T> class Decorator>
      struct is_plchldr_to_temp<arg<I, Decorator<First, BaseType ...> > > : is_plchldr_to_temp<arg<I, typename First::basic_type> >
    {};
#endif

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
        constexpr expr(ArgType1 const& first_operand, ArgType2 const& second_operand)
            :
#if( (!defined(CXX11_ENABLED)))
      first_operand(first_operand),
	second_operand(second_operand)
#else
            first_operand{first_operand},
            second_operand{second_operand}
#endif
            {}
      constexpr expr(){}
        ArgType1 const first_operand;
        ArgType2 const second_operand;
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_plus : public expr<ArgType1, ArgType2>{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_plus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
	constexpr expr_plus(){};
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_minus : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_minus(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
	constexpr expr_minus(){}
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_times : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_times(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
	constexpr expr_times(){}
    };

    template <typename ArgType1, typename ArgType2>
    struct expr_divide : public expr<ArgType1, ArgType2 >{
        typedef expr<ArgType1, ArgType2> super;
        GT_FUNCTION
        constexpr expr_divide(ArgType1 const& first_operand, ArgType2 const& second_operand):super(first_operand, second_operand){}
	constexpr expr_divide(){}
   };

#ifdef CXX11_ENABLED
    namespace expressions{
        template<typename ArgType1, typename ArgType2>
        GT_FUNCTION
        constexpr expr_plus<ArgType1, ArgType2 >  operator + (ArgType1 arg1, ArgType2 arg2){return expr_plus<ArgType1, ArgType2 >(arg1, arg2);}

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
