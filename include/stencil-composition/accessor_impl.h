#pragma once

#include "../storage/storage.h"
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

namespace gridtools {


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
        GT_FUNCTION
        static void apply(int_t* offset, X ... x)
            {
                offset[ID]=initialize<ID>(x...);
                initialize_all<ID-1>::apply(offset, x...);
            }
    };

    template<>
    struct initialize_all<0>{

        template <typename ... X>
        GT_FUNCTION
        static void apply(int_t* offset, X ... x)
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

    /**@brief method for initializing the offsets in the placeholder
       Version valid for three dimension
       \param x is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (x.value) and the dimension index (X::direction)
       \param y is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (y.value) and the dimension index (Y::direction)
       \param z is an instance of the \ref gridtools::enumtype::Dimension class, which contains the offset (z.value) and the dimension index (Z::direction)
    */
    template <ushort_t N, typename X, typename Y, typename Z, typename T>
    GT_FUNCTION
    constexpr int_t initialize(X x, Y y, Z z, T t)
    {
        return X::direction==N? x.value : Y::direction==N? y.value : Z::direction==N? z.value : T::direction==N? t.value : 0;
    }
#endif

    //forward declaration
    template< int_t Index, int_t Dimension >
    struct offset_tuple;

    namespace enumtype{
        template <ushort_t>
        struct Dimension;
    }

    template <uint_t I, typename T>
    struct arg;

    /**
     * @brief Type to be used in elementary stencil functions to specify argument mapping and ranges
     *
     One accessor consists substantially of an array of offsets (runtime values), a range and an index (copmpile-time constants). The latter is used to distinguish the types of two different accessors,
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
    template <uint_t I, typename Range, ushort_t Dim >
    struct accessor_base  {

        //typedef useful when unnecessary indirections are used
        typedef accessor_base<I, Range, Dim> type;
        template <uint_t II, typename R, ushort_t D>
        friend std::ostream& operator<<(std::ostream& s, accessor_base<II,R,D> const& x);
        typedef accessor_base<I,Range,Dim> base_t;
        static const ushort_t n_dim=Dim;

        typedef static_uint<I> index_type;
        typedef Range range_type;


        /**@brief Default constructor
           NOTE: the following constructor when used with the brace initializer produces with nvcc a considerable amount of extra instructions (gcc 4.8.2), and degrades the performances (which is probably a compiler bug, I couldn't reproduce it on a small test).*/
        GT_FUNCTION
        constexpr explicit accessor_base():m_offsets() {}

#ifdef CXX11_ENABLED
        //move ctor
        GT_FUNCTION
        constexpr accessor_base(const type && other) : m_offsets(other.m_offsets){}

        //move ctor from another accessor_base with different index
        template<ushort_t OtherIndex>
        GT_FUNCTION
        constexpr accessor_base(accessor_base<OtherIndex, Range, Dim>&& other) :
            m_offsets(other.offsets()) {}
#else
        //copy ctor
        GT_FUNCTION
        constexpr accessor_base(const type & other) : m_offsets(other.m_offsets){}

        //copy ctor from another accessor_base with different index
        template<uint_t OtherIndex>
        GT_FUNCTION
        constexpr accessor_base(const accessor_base<OtherIndex, Range, Dim> & other) :
            m_offsets(other.offsets()){}
#endif

        //ctor with one argument have to provide specific arguments in order to avoid ambiguous instantiation
        // by the compiler
        template<ushort_t Idx>
        GT_FUNCTION
        constexpr accessor_base (enumtype::Dimension<Idx> const& x ): m_offsets(x) {}

        GT_FUNCTION
        constexpr accessor_base (const int_t x ): m_offsets(x) {}


        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::Dimension' is a
           language keyword used at the interface level.
        */
#ifdef CXX11_ENABLED
        template <typename... Whatever>
        GT_FUNCTION
        constexpr accessor_base ( Whatever... x) : m_offsets(x...)
            {
                GRIDTOOLS_STATIC_ASSERT(sizeof...(x)<=n_dim, "the number of arguments passed to the offset_tuple constructor exceeds the number of space dimensions of the storage")
            }
#else
        template<typename X, typename Y, typename Z, typename T>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y, Z z, T t ): m_offsets(x,y,z,y)
            {
            }

        template<typename X, typename Y, typename Z>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y, Z z ): m_offsets(x,y,z)
            {
            }

        template<typename X, typename Y>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y ): m_offsets(x,y) {}
#endif

        static  void info() {
            std::cout << "Arg_type storage with index " << I << " and range " << Range() << " ";
        }

        template<short_t Idx>
        constexpr bool end() const {return true;}

        template<short_t Idx>
        GT_FUNCTION
        constexpr
        int_t get() const {
            GRIDTOOLS_STATIC_ASSERT(Idx<=n_dim, "requested accessor index larger than the available dimensions")
            GRIDTOOLS_STATIC_ASSERT(Idx>=0, "requested accessor index lower than zero")
            return m_offsets.template get<Idx>();
        }

        GT_FUNCTION
        constexpr const offset_tuple<n_dim, n_dim>& offsets() const { return m_offsets;}

    private:

        offset_tuple<n_dim, n_dim> m_offsets;

// #ifdef CXX11_ENABLED
// #ifndef __CUDACC__
// static const constexpr char a[]={"arg "};
// typedef string<print, static_string<a>, static_int<I> > to_string;
// #endif
// #endif
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
    template< uint_t Index, int_t Dimension >
    struct offset_tuple : public offset_tuple<Index-1, Dimension>
    {
        static const int_t n_dim=Dimension;

        typedef offset_tuple<Index-1, Dimension> super;
        static const ushort_t n_args=super::n_args+1;

#ifdef CXX11_ENABLED

        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base class (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in order. No check is done on the order*/
        template <typename... Whatever>
        GT_FUNCTION
            constexpr offset_tuple ( int const& t, Whatever const& ... x): super( x... ), m_offset(t) {
        }

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::Dimension' is a
           language keyword used at the interface level.
        */
        template <ushort_t Idx, typename... Whatever>
        GT_FUNCTION
        constexpr offset_tuple ( enumtype::Dimension<Idx> const& t, Whatever const&... x):
            super( t, x... ), m_offset(initialize<super::n_dim-n_args+1>(t, x...))
            {
                //this constructor should be a constexpr one (waiting for future standards (C++14) for that)
                //m_offset[n_args-1] = initialize<n_args>(t, x...);
            }
#else
        /**@brief constructor taking an integer as the first argument, and then other optional arguments.
           The integer gets assigned to the current extra dimension and the other arguments are passed to the base class (in order to get assigned to the other dimensions).
           When this constructor is used all the arguments have to be specified and passed to the function call in order. No check is done on the order*/
        GT_FUNCTION
        offset_tuple ( int const& i, int const& j, int const& k): super( j, k ), m_offset(i) {
        }
        GT_FUNCTION
        offset_tuple ( int const& i, int const& j): super( j ), m_offset(i) {
        }
        GT_FUNCTION
        offset_tuple ( int const& i): m_offset(i) {
        }

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::Dimension' is a
           language keyword used at the interface level.
        */
        template <ushort_t Idx1, ushort_t Idx2, ushort_t Idx3, ushort_t Idx4 >
        GT_FUNCTION
        offset_tuple ( enumtype::Dimension<Idx1> const& t, enumtype::Dimension<Idx2> const& u, enumtype::Dimension<Idx3> const& v,  enumtype::Dimension<Idx4> const& h ): super(t, u, v, h), m_offset(initialize<super::n_dim-n_args+1>(t, u, v, h))
            {
                //base_t::m_offset[n_args-1] = initialize<n_args>(t, u, v);
            }

        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::Dimension' is a
           language keyword used at the interface level.
        */
        template <ushort_t Idx1, ushort_t Idx2, ushort_t Idx3 >
        GT_FUNCTION
        offset_tuple ( enumtype::Dimension<Idx1> const& t, enumtype::Dimension<Idx2> const& u, enumtype::Dimension<Idx3> const& v ): super(t, u, v), m_offset(initialize<super::n_dim-n_args+1>(t, u, v))
            {
                //base_t::m_offset[n_args-1] = initialize<n_args>(t, u, v);
            }
        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::Dimension' is a
           language keyword used at the interface level.
        */
        template <ushort_t Idx1, ushort_t Idx2 >
        GT_FUNCTION
        offset_tuple ( enumtype::Dimension<Idx1> const& t, enumtype::Dimension<Idx2> const& u ): super(t,u), m_offset(initialize<super::n_dim-n_args+1>(t, u))
            {
                //base_t::m_offset[n_args-1] = initialize<n_args>(t, u);
            }
        /**@brief constructor taking the Dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'enumtype::Dimension' is a
           language keyword used at the interface level.
        */
        template <ushort_t Idx >
        GT_FUNCTION
        offset_tuple ( enumtype::Dimension<Idx> const& t ) : super(t), m_offset(initialize<super::n_dim-n_args+1>(t))
            {
                //base_t::m_offset[n_args-1] = initialize<n_args>(t);
            }
#endif

        //initializes recursively all the offsets to 0
        GT_FUNCTION
        constexpr offset_tuple ( ):
            super( ), m_offset(0)
            {
                //base_t::m_offset[n_args-1] = 0;
            }

        // /**@brief returns the offset tuple (i.e. this instance)*/
        // offset_tuple<ArgType> const& offset() const {return *this;}

        template<short_t Idx>
        constexpr bool end() const {return Idx==n_args-1? false : super::template end<Idx>();}

        /**@brief returns the offset at a specific index Idx*/
        template<short_t Idx>
        /**@brief returns the offset array*/
        GT_FUNCTION
        constexpr
        int_t get() const {
            //NOTE: this should be a constexpr whenever m_offset is a static const
            //this might not be compile-time efficient for large indexes,
            //because both taken and not taken branches are compiled. boost::mpl::eval_if would be better.
            return Idx==n_args-1? m_offset : super::template get<Idx>();

        }

    protected:
        int_t m_offset;
    };

    //specialization
    template< int_t Dimension >
    struct offset_tuple<0, Dimension>
    {
        static const int_t n_dim=Dimension;
        #ifdef CXX11_ENABLED
        template <typename... Whatever>
        GT_FUNCTION
        constexpr offset_tuple ( Whatever... x) {}

        //copy ctor
        GT_FUNCTION
        constexpr offset_tuple (const offset_tuple<0, Dimension>& other) {}
#else
        template <typename X, typename Y, typename Z,  typename T>
        GT_FUNCTION
        constexpr offset_tuple ( X x, Y y, Z z, T t ){}

        template <typename X, typename Y, typename Z>
        GT_FUNCTION
        constexpr offset_tuple ( X x, Y y, Z z ){}

        template <typename X>
        GT_FUNCTION
        constexpr offset_tuple ( X x ){}

        template <typename X, typename Y>
        GT_FUNCTION
        constexpr offset_tuple ( X x, Y y ){}
#endif

        GT_FUNCTION
        constexpr offset_tuple(){}
        static const ushort_t n_args=0;

        template<short_t Idx>
        GT_FUNCTION
        constexpr int_t get() const { return 0;}
    };

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
    template <uint_t I, typename T, typename U, short_t Dim>
    struct is_plchldr_to_temp<arg<I, base_storage< T, U,  true, Dim> > > : boost::true_type
    {};

    /**
     * Struct to test if an argument is a placeholder to a temporary storage - Specialization yielding false
     */
    template <uint_t I, typename T, typename U, short_t Dim>
    struct is_plchldr_to_temp<arg<I, base_storage<  T, U,false, Dim> > > : boost::false_type
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
     here the decorator is the dimension extension, \ref gridtools::data_field
    */
    template <uint_t I, typename First, typename ... BaseType, template <typename ... T> class Decorator>
    struct is_plchldr_to_temp<arg<I, Decorator<First, BaseType ...> > > : is_plchldr_to_temp<arg<I, typename First::basic_type> >
    {};

#else

    template <uint_t I, typename First, typename B2, typename B3, template <typename  T1, typename  T2, typename  T3> class Decorator>
    struct is_plchldr_to_temp<arg<I, Decorator<First, B2, B3> > > : is_plchldr_to_temp<arg<I, typename First::basic_type> >
    {};

#endif

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for offset_tuple
     * @return ostream
     */
    template <uint_t I, typename R, ushort_t D>
    std::ostream& operator<<(std::ostream& s, accessor_base<I,R,D> const& x) {
        s << "[ offset_tuple< " << I
                 << ", " << R()
                 << ", " << D
                 // << " (" << x.i()
                 // << ", " << x.j()
                 // << ", " << x.k()
                 <<" ) > m_offset: {";

        for (int i=0; i<x.n_dim-1; ++i) {
            s << x.m_offset[i] << ", ";
        }
        s << x.m_offset[x.n_dim-1] << "} ]";
        return s;
    }

    /**
     * Printing type information for debug purposes
     * @param s The ostream
     * @param n/a Type selector for offset_tuple
     * @return ostream
     */
    template <uint_t I, typename R>
    std::ostream& operator<<(std::ostream& s, arg<I,no_storage_type_yet<R> > const&) {
        return s << "[ arg< " << I
                 << ", temporary<something>" << " > ]";
    }

    /**
     * Printing type information for debug purposes
<     * @param s The ostream
     * @param n/a Type selector for arg to a NON temp
     * @return ostream
     */
    template <uint_t I, typename R>
    std::ostream& operator<<(std::ostream& s, arg<I,R> const&) {
        return s << "[ arg< " << I
                 << ", NON TEMP" << " > ]";
    }


    /**
       \addtogroup specializations Specializations
       @{
    */
    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>  > : public boost::true_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

    template <typename T, typename U, short_t Dim>
    struct is_storage<base_storage<T,U,true, Dim>  *  > : public boost::false_type
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

    template <typename T, typename U, short_t Dim>
    struct is_storage<base_storage<T,U,false, Dim>  *  > : public boost::true_type
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
#else

    //Decorator is the integrator
    template <typename First, typename B2, typename  B3 , template <typename T1, typename T2, typename T3> class Decorator >
    struct is_storage<Decorator<First, B2, B3>  *  > : public is_storage<typename First::basic_type*>
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

#endif

    //Decorator is the integrator
    template <typename BaseType , template <typename T, ushort_t O> class Decorator, ushort_t Order >
    struct is_storage<Decorator<BaseType, Order>  *  > : public is_storage<typename BaseType::basic_type*>
    { /*BOOST_MPL_ASSERT( (boost::mpl::bool_<false>) );*/};

/**
   @}
*/

} // namespace gridtools
