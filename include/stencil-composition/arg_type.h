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
#ifdef CXX11_ENABLED
#include "expressions.h"
#endif

/**
   @file
   @brief File containing the definition of the placeholders used to address the storage from whithin the functors.
   A placeholder is an implementation of the proxy design pattern for the storage class, i.e. it is a light object used in place of the storage when defining the high level computations,
   and it will be bound later on with a specific instantiation of a storage class.

   Two different types of placeholders are considered:
   - arg represents the storage in the body of the main function, and it gets lazily assigned to a real storage.
   - arg_type represents the storage inside the functor struct containing a Do method. It can be instantiated directly in the Do method, or it might be a constant expression instantiated outside the functor scope and with static duration.
*/

namespace gridtools {

    /**
     * @brief Type to indicate that the type is not decided yet
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

    /**
       @brief stream operator, for debugging purpose
    */
    template <typename RST>
    std::ostream& operator<<(std::ostream& s, no_storage_type_yet<RST>) {
        return s << "no_storage_type_yet<" << RST() << ">" ;
    }

    /**
       \addtogroup specializations Specializations
       @{
    */
    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>  > : public boost::true_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};



    template <enumtype::backend X, typename T, typename U, short_t Dim>
    struct is_storage<base_storage<X,T,U,true, Dim>  *  > : public boost::false_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};



    template <enumtype::backend X, typename T, typename U, short_t Dim>
    struct is_storage<base_storage<X,T,U,false, Dim>  *  > : public boost::true_type
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

/**
   @}
*/
    /** @brief binding between the placeholder (\tparam ArgType) and the storage (\tparam Storage)*/
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
        /**
           @brief The following struct defines one specific component of a field
           It contains a direction (compile time constant, specifying the ID of the component), and a value
           (runtime value, which is storing the offset in the given direction).
           As everything what is inside the enumtype namespace, the Dimension keyword is supposed to be used at the application interface level.
        */
        template <ushort_t Coordinate>
        struct Dimension{

	    template <typename IntType>
            GT_FUNCTION
            constexpr Dimension(IntType val) : value
#if( (!defined(CXX11_ENABLED)) )
                                             (val)
#else
                {val}
#endif
                {}

	    GT_FUNCTION
	    constexpr Dimension(Dimension const& other):value(other.value){}

            static const ushort_t direction=Coordinate;
            int_t value;
	    struct Index{
		GT_FUNCTION
		Index(){}
		GT_FUNCTION
		Index(Index const&){}

		typedef Dimension<Coordinate> super;
	    };

	private:
	    Dimension();
        };

        /**Aliases for the first three dimensions (x,y,z)*/
        typedef Dimension<0> x;
        typedef Dimension<1> y;
        typedef Dimension<2> z;

    }

    /**@brief method for initializing the offsets in the placeholder
       Version valid for one dimension
       \param x is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (x.value) and the dimension index (X::direction)
    */
    template <ushort_t N, typename X>
    GT_FUNCTION
    constexpr int_t initialize( X x )
    {
        return (X::direction==N? x.value : 0);
    }

#ifdef CXX11_ENABLED
    /**@brief method for initializing the offsets in the placeholder
       Version valid for arbitrary dimension
       \param x is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (x.value) and the dimension index (X::direction)
       \param rest are the remaining arguments, which get considered one at a time in by means of recursive calls
    */
    template <ushort_t N, typename X, typename ... Rest>
    GT_FUNCTION
    constexpr int_t initialize(X x, Rest ... rest )
    {
        return X::direction==N? x.value : initialize<N>(rest...);
    }

    template<ushort_t ID>
    struct initialize_all{
	template <typename ... X>
	static void constexpr apply(int_t* offset, X ... x)
	    {
		offset[ID]=initialize<ID>(x...);
		initialize_all<ID-1>::apply(offset, x...);
	    }
    };

    template<>
    struct initialize_all<0>{
	template <typename ... X>
	static void constexpr apply(int_t* offset, X ... x)
	    {
		offset[0]=initialize<0>(x...);
	    }
    };
#else

    /**@brief method for initializing the offsets in the placeholder
       Version valid for two dimension
       \param x is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (x.value) and the dimension index (X::direction)
       \param y is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (y.value) and the dimension index (Y::direction)
    */
    template <ushort_t N, typename X, typename Y>
    GT_FUNCTION
    constexpr int_t initialize(X x, Y y)
    {
        return X::direction==N? x.value : Y::direction==N? y.value : 0;
    }

    /**@brief method for initializing the offsets in the placeholder
       Version valid for three dimension
       \param x is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (x.value) and the dimension index (X::direction)
       \param y is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (y.value) and the dimension index (Y::direction)
       \param z is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (z.value) and the dimension index (Z::direction)
    */
    template <ushort_t N, typename X, typename Y, typename Z>
    GT_FUNCTION
    constexpr int_t initialize(X x, Y y, Z z)
    {
        return X::direction==N? x.value : Y::direction==N? y.value : Z::direction==N? z.value : 0;
    }
#endif

    /**
     * @brief Type to be used in elementary stencil functions to specify argument mapping and ranges
     *
     One arg_type consists substantially of an array of offsets (runtime values), a range and an index (copmpile-time constants). The latter is used to distinguish the types of two different arg_types,
     while the offsets are used to calculate the final memory address to be accessed by the stencil in \ref gridtools::iterate_domain.
     * The class also provides the interface for accessing data in the function body.
     The interfaces available to specify the offset in each dimension are covered in the following example, supposing that we have to specify the offsets of a 3D field V:
     - specify three offsets: in this case the order matters, the three arguments represent the offset in the  i, j, k directions respectively.
     \verbatim
     V(1,0,-3)
     \endverbatim
     - specify only some components: in this case the order is arbitrary and the missing components have offset zero;
     \verbatim
     V(z(-3),x(1))
     \endverbatim
     *
     * @tparam I Index of the argument in the function argument list
     * @tparam Range Bounds over which the function access the argument
     */
    template <uint_t I, typename Range=range<0,0,0,0>, ushort_t dimension=3 >
    struct arg_type   {

        int_t m_offset[dimension]
#ifdef CXX11_ENABLED
        ={0}
#endif
            ;

        typedef static_uint<I> index_type;
        typedef Range range_type;

        /**
           @brief Constructor with three integer offsets
           \param i the offset in x direction
           \param j the offset in y direction
           \param k the offset in z direction
           NOTE: templating on the int type, because if we use int_t directly, and if int_t is different from int, then the user would have to explicitly specify the cast to int_t*/
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


#if defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9) )
#warning "Obsolete version of the GCC compiler"
        // GCC compiler bug solved in versions 4.9+, Clang is OK, other copmilers were not tested
        // while waiting for an update in nvcc (which is not supporting gcc 4.9 at present)
        // we implement a suboptimal solution
        /**
           @brief Constructor with three dimensions
           The arguments are of type \ref gridtools::enumtype::Dimension , containing an offset (run time) and the dimension index (compile time)
           \param x1 the first Dimension, can be in x,y,z direction (i.e. an instance of either Dimension<0>, Dimension<1> or Dimension<2>, which are conveniently aliased with x, y, z)
           \param x2 the second Dimension, can be in x,y,z direction
           \param x3 the third Dimension, can be in x,y,z direction
         */
        template <typename X1, typename X2, typename X3 >
        GT_FUNCTION
        arg_type ( X1 x, X2 y, X3 z)
#if( (!defined(CXX11_ENABLED)))
            {
                m_offset[0]=initialize<0>(x,y,z);
                m_offset[1]=initialize<1>(x,y,z);
                m_offset[2]=initialize<2>(x,y,z);
                GRIDTOOLS_STATIC_ASSERT(X1::direction<3 && X2::direction<3 && X3::direction<3, "You specified a dimension index exceeding the total number of dimensions");
            }
#else
        :m_offset{initialize<0>(x,y,z), initialize<1>(x,y,z), initialize<2>(x,y,z)}{
            GRIDTOOLS_STATIC_ASSERT(X1::direction<3 && X2::direction<3 && X3::direction<3, "You specified a dimension index exceeding the total number of dimensions");
        }
#endif

        /**
           @brief Constructor with two dimensions
           The arguments are of type \ref gridtools::enumtype::Dimension , containing an offset (run time) and the dimension index (compile time)
           \param x1 the first Dimension, can be in x,y,z direction (i.e. an instance of either Dimension<0>, Dimension<1> or Dimension<2>, which are conveniently aliased with x, y, z)
           \param x2 the second Dimension, can be in x,y,z direction
         */
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

        /**
           @brief Constructor with one dimension
           The arguments are of type \ref gridtools::enumtype::Dimension , containing an offset (run time) and the dimension index (compile time)
           \param x1 the first Dimension, can be in x,y,z direction (i.e. an instance of either Dimension<0>, Dimension<1> or Dimension<2>, which are conveniently aliased with x, y, z)
         */
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

        /**
           @brief Constructor with arbitrary dimension
           The arguments are of type \ref gridtools::enumtype::Dimension , containing an offset (run time) and the dimension index (compile time)
           \param x the Dimensions, can be in x,y,z direction (i.e. an instance of either Dimension<0>, Dimension<1> or Dimension<2>, which are conveniently aliased with x, y, z)
         */
        template <typename... X >
        GT_FUNCTION
        /*constexpr*/ arg_type ( X... x)//:m_offset{initialize<0>(x...), initialize<1>(x...), initialize<2>(x...)}
	    {
		/**how to initialize before the constructor body?
		   Because of that this isn't a constexpr constructor, but it should be*/
		/*TODO: initialize only the dimensions affected*/
		initialize_all<dimension-1>::apply(m_offset, x...);
		//      if you get a compiler error here, use the version above
	    }
#endif //__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)


        /**@brief Default constructor
           NOTE: the following constructor when used with the brace initializer produces with nvcc a considerable amount of extra instructions (gcc 4.8.2), and degrades the performances (which is probably a compiler bug, I couldn't reproduce it on a small test).*/
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

        /**@brief returns the offset array*/
        GT_FUNCTION
        constexpr int_t const* offset() const {
            return m_offset;
        }

        /**@brief returns the offset in x direction*/
        GT_FUNCTION
        constexpr int_t i() const {
            return m_offset[0];
        }

        /**@brief returns the offset in y direction*/
        GT_FUNCTION
        constexpr int_t j() const {
            return m_offset[1];
        }

        /**@brief returns the offset in z direction*/
        GT_FUNCTION
        constexpr int_t k() const {
            return m_offset[2];
        }

        /**@brief returns a copy of the arg_type with all offsets set to zero*/
        GT_FUNCTION
#ifdef CXX11_ENABLED
        static constexpr  arg_type<I>&& center() {
            return std::move(arg_type<I>());
        }
#else
        static constexpr  arg_type<I> center() {
            return arg_type<I>();
        }
#endif

        /**@brief returns the pointer to the first element of the array of offsets*/
        GT_FUNCTION
        constexpr int_t const* offset_ptr() const {
            return &m_offset[0];
        }

        /**returns a new arg_type where the offsets are the sum of the current offsets plus the values specified via the argument*/
        GT_FUNCTION
#ifdef CXX11_ENABLED
        constexpr arg_type<I>&& plus(int_t _i, int_t _j, int_t _k) const {
            return std::move(arg_type<I>(i()+_i, j()+_j, k()+_k));
        }
#else
        constexpr arg_type<I> plus(int_t _i, int_t _j, int_t _k) const {
            return arg_type<I>(i()+_i, j()+_j, k()+_k);
        }
#endif

        static  void info() {
            std::cout << "Arg_type storage with index " << I << " and range " << Range() << " ";
        }

        // Methods to stop the recursion when dealing with extra dimensions
	/**TODO: this should not be hardcoded*/
        static const ushort_t n_args=2;// max space dimensions
        static const ushort_t extra_args=0;// extra dimensions

        template<short_t idx>
        GT_FUNCTION
        int_t n() const {//stop recursion
            printf("The dimension you are trying to access exceeds the number of dimensions by %d.\n ", idx+1);
            exit (-1);
        }

#ifdef CXX11_ENABLED
#ifndef __CUDACC__
	static const constexpr char a[]={"arg "};
	typedef string<print, static_string<a>, static_int<I> > to_string;
#endif
#endif
    };


//################################################################################
//                              Multidimensional Fields
//################################################################################

    /**@brief this is a decorator of the arg_type, which is matching the extra dimensions
       \param n_args is the current ID of the extra dimension
       \param index_type is the index of the storage type

       EXAMPLE:

       Possible interfaces to access one extra dimension T (say, temperature) at offset -1 of a 'velocity' field V are the following:
       - specify it with the first integer argument (the arguments after the first define the offsets of the 3D fields and can be in any of the form described in gridtools::arg_type)
       \verbatim
       V(-1, x(1), z(-3));
       \endverbatim
       - specify explicitly the dimension: in this case the order of the arguments is arbitrary:
       \verbatim
       typedef Dimension<4> T;
       V(x(1), z(-3), T(-1))
       \endverbatim

       Note that if no value is specified for the extra dimension a zero offset is implicitly assumed.
    */
    template< class ArgType>
    struct arg_decorator : public ArgType{

        typedef ArgType super;
        static const ushort_t n_args=super::n_args+1;
        static const ushort_t extra_args=super::extra_args+1;
        typedef typename super::index_type index_type;

#ifdef CXX11_ENABLED
        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base class (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in order. No check is done on the order*/
        template <typename IntType, typename... Whatever>
        GT_FUNCTION
        arg_decorator ( IntType const& t, Whatever... x): super( x... ) {
            m_offset=t;
        }

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::Dimension' is a
           language keyword used at the interface level.
        */
        template <ushort_t Idx, typename... Whatever>
        GT_FUNCTION
        arg_decorator ( enumtype::Dimension<Idx> const& t, Whatever... x): super( x... ) {

            //if the following check is not true, you specified an extra index exceeding the dimenison of the field
            //BOOST_STATIC_ASSERT(enumtype::Dimension<Idx>::direction<=n_args);

            if(enumtype::Dimension<Idx>::direction==n_args)
            {
		// printf("offset %d was specified to be %d \n", n_args, t.value);
                m_offset=t.value;
            }
            else
            {
		// printf("no offset was specified for extra dimension %d ( != %d) \n", t.direction, n_args);
                m_offset=0;
            }
        }

        /**@brief fallback constructor, when the others are not taken, meaning that no offset for the extra dimension was specified, simply forwards the call to the constructor of the base class.*/
        template <typename... Whatever>
        GT_FUNCTION
        arg_decorator ( Whatever... x ): super( x... ) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(x)<=n_args, "the number of arguments passed to the arg_type constructor exceeds the number of space dimensions of the storage");
            // printf("no offsets for extra dimension was specified (but there are %d) \n", n_args);
            m_offset=0;
        }//just forward

// #else //CXX11_ENABLED
// whatever not compiling
#endif

        /** @brief usage: n<3>() returns the offset of extra dimension 3
            loops recursively over the children, decreasing each time the index, until it has reached the dimension matching the index specified as template argument.
            Note that here the offset we are talking about here looks the same as the offsets for the arg_type, but it implies actually a change of the base storage pointer.
	    TODO change this stupid name
        */
	template<short_t idx>
	GT_FUNCTION
	short_t n() const {//recursively travel the list of offsets
	    //the following assert cannot be compile time, since a version with idx=-1 may indeed be compiled (at the end of the template recursion), but should never be executed
#ifndef __CUDACC__
	    assert(idx>0);
#endif
	    //BOOST_STATIC_ASSERT( index>0 );
	    // printf("index to the n method:%d \n", index);
	    GRIDTOOLS_STATIC_ASSERT( idx<=n_args, "the index passed as template argument is too large" );
	    //this might not be compile-time efficient for large indexes,
	    //because both taken and not taken branches are compiled. boost::mpl::if would be better.
            return idx==1? m_offset : super::template n<idx-1>();
	    // return static_if<idx==1>::apply( m_offset,super::template n<idx-1>());
	}

	//std::string m_arg_string(m_offset+ std::string(", ") +super::arg_string);

    private:
        short_t m_offset;
    };


    /**@brief Convenient syntactic sugar for specifying an extended-width storage with size 'Number' (similar to currying)
       The extra width is specified using concatenation, e.g. extending arg_type with 2 extra data fields is obtained by doing
       \verbatim
       arg_decorator<arg_decorator<arg_type>>
       \endverbatim
       The same result is achieved using the arg_extend struct with
       \verbatim
       arg_extend<arg_type, 2>
       \endverbatim
     */
    template < typename ArgType, uint_t Number=2>
    struct arg_extend{
        typedef arg_decorator<typename arg_extend<ArgType, Number-1>::type>  type;
    };

    /**@brief specialization to stop the recursion*/
    template<typename ArgType>
    struct arg_extend<ArgType, 0>{typedef ArgType type;};

#ifdef CXX11_ENABLED
/**this struct allows the specification of SOME of the arguments before instantiating the arg_type.
   It is a language keyword.
*/
template <typename Callable, typename ... Known>
struct alias{

    /**@brief constructor
       \param args are the offsets which are already known*/
    template<typename ... Args>
    GT_FUNCTION
    constexpr alias( Args/*&&*/ ... args ): m_knowns{args ...} {
    }

    typedef boost::mpl::vector<Known...> dim_vector;

    /** @brief operator calls the constructor of the arg_type
	\param unknowns are the parameters which were not known beforehand. They might be instances of
	the enumtype::Dimension class. Together with the m_knowns offsets form the arguments to be
	passed to the Callable functor (which is normally an instance of arg_type)
     */
    template<typename ... Unknowns>
    GT_FUNCTION
    Callable/*&&*/ operator() ( Unknowns/*&&*/ ... unknowns  )
        {
	    return Callable(enumtype::Dimension<Known::direction> (m_knowns[boost::mpl::find<dim_vector, Known>::type::pos::value]) ... , unknowns ...);}

private:
    //store the list of offsets which are already known on an array
    int_t m_knowns [sizeof...(Known)];
};

#endif

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

    /**
     * Struct to test if an argument is a placeholder to a temporary storage - Specialization yielding true
     */
    template <uint_t I, enumtype::backend X, typename T, typename U, short_t Dim>
    struct is_plchldr_to_temp<arg<I, base_storage<X, T, U,  true, Dim> > > : boost::true_type
    {};

    /**
     * Struct to test if an argument is a placeholder to a temporary storage - Specialization yielding false
     */
    template <uint_t I, enumtype::backend X, typename T, typename U, short_t Dim>
    struct is_plchldr_to_temp<arg<I, base_storage< X, T, U,false, Dim> > > : boost::false_type
    {};

    /**
     * Struct to test if an argument is a temporary no_storage_type_yet - Specialization for a decorator of the storage class, falls back on the original class type
     here the decorator is the \ref gridtools::storage
    */
    template <uint_t I, typename BaseType, template <typename T> class Decorator>
    struct is_plchldr_to_temp<arg<I, Decorator<BaseType> > > : is_plchldr_to_temp<arg<I, typename BaseType::basic_type> >
    {};

#ifdef CXX11_ENABLED

    /**
     * Struct to test if an argument is a temporary no_storage_type_yet - Specialization for a decorator of the storage class, falls back on the original class type
     here the decorator is the dimension extension, \ref gridtools::extend_dim
    */
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

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for arg_type
     * @return ostream
     */
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
} // namespace gridtools
