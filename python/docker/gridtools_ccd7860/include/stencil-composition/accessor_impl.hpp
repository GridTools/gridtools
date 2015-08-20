#pragma once

#include "storage/storage.hpp"
#include "common/layout_map.hpp"
#include "range.hpp"
#include <boost/type_traits/integral_constant.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <vector>
#include "common/is_temporary_storage.hpp"
#include "stencil-composition/offset_tuple.hpp"
#ifdef CXX11_ENABLED
#include "expressions.hpp"
#endif

namespace gridtools {

    //forward declaration
    template< int_t  Index, int_t NDim >
    struct offset_tuple;

    template <ushort_t>
    struct dimension;

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

#if defined( CXX11_ENABLED ) && ! defined(__CUDACC__)
        //move ctor
        GT_FUNCTION
        constexpr accessor_base(const type && other) : m_offsets(other.m_offsets){}

        //move ctor from another accessor_base with different index
        template<uint_t OtherIndex>
        GT_FUNCTION
        constexpr accessor_base(accessor_base<OtherIndex, Range, Dim>&& other) :
            m_offsets(other.offsets()) {}
#endif
        //copy ctor
        GT_FUNCTION
        constexpr accessor_base( type const& other) : m_offsets(other.m_offsets){}

        //copy ctor from another accessor_base with different index
        template<uint_t OtherIndex>
        GT_FUNCTION
        constexpr accessor_base(const accessor_base<OtherIndex, Range, Dim> & other) :
            m_offsets(other.offsets()){}

        //ctor with one argument have to provide specific arguments in order to avoid ambiguous instantiation
        // by the compiler
        template<uint_t Idx>
        GT_FUNCTION
        constexpr accessor_base (dimension<Idx> const& x ): m_offsets(x) {}

        GT_FUNCTION
        constexpr accessor_base (const int_t x ): m_offsets(x) {}


        /**@brief constructor taking the dimension class as argument.
           This allows to specify the extra arguments out of order. Note that 'dimension' is a
           language keyword used at the interface level.
        */
#if defined( CXX11_ENABLED ) && ! defined(__CUDACC__) //cuda messing up
        template <typename... Whatever>
        GT_FUNCTION
        constexpr accessor_base ( Whatever... x) : m_offsets( x...)
        {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(x)<=n_dim, "the number of arguments passed to the offset_tuple constructor exceeds the number of space dimensions of the storage. Check that you are not accessing a non existing dimension, or increase the dimension D of the accessor (accessor<Id, range, D>)");
        }
#else
        template<typename X, typename Y, typename Z, typename T>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y, Z z, T t ): m_offsets(x,y,z,y)
        {}

        template<typename X, typename Y, typename Z>
        GT_FUNCTION
        constexpr accessor_base (X x, Y y, Z z ): m_offsets(x,y,z)
        {}

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
            GRIDTOOLS_STATIC_ASSERT(Idx<=n_dim, "requested accessor index larger than the available dimensions");
            GRIDTOOLS_STATIC_ASSERT(Idx>=0, "requested accessor index lower than zero");
            return m_offsets.template get<Idx>();
        }

        GT_FUNCTION
        constexpr const offset_tuple<n_dim, n_dim>& offsets() const { return m_offsets;}

    private:

        offset_tuple<n_dim, n_dim> m_offsets;
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
     * Struct to test if an argument is a temporary
     no_storage_type_yet - Specialization for a decorator of the
     storage class, falls back on the original class type here the
     decorator is the \ref gridtools::storage
    */
    template <uint_t I, typename BaseType, template <typename T> class Decorator>
    struct is_plchldr_to_temp<arg<I, Decorator<BaseType> > > : is_plchldr_to_temp<arg<I, typename BaseType::basic_type> >
    {};

#ifdef CXX11_ENABLED

    /**
     * Struct to test if an argument is a temporary
     no_storage_type_yet - Specialization for a decorator of the
     storage class, falls back on the original class type here the
     decorator is the dimension extension, \ref gridtools::data_field
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
    {};

    template <typename T, typename U, short_t Dim>
    struct is_storage<base_storage<T,U,true, Dim>  *  > : public boost::false_type
    {};

    template <typename T, typename U, short_t Dim>
    struct is_storage<base_storage<T,U,false, Dim>  *  > : public boost::true_type
    {};

    template <typename U>
    struct is_storage<no_storage_type_yet<U>  *  > : public boost::false_type
    {};

    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>* > : public boost::true_type
    {};

    template <typename U>
    struct is_temporary_storage<no_storage_type_yet<U>& > : public boost::true_type
    {};

    //Decorator is the storage
    template <typename BaseType , template <typename T> class Decorator >
    struct is_storage<Decorator<BaseType>  *  > : public is_storage<typename BaseType::basic_type*>
    {};

#ifdef CXX11_ENABLED
    //Decorator is the integrator
    template <typename First, typename ... BaseType , template <typename ... T> class Decorator >
    struct is_storage<Decorator<First, BaseType...>  *  > : public is_storage<typename First::basic_type*>
    {};
#else

    //Decorator is the integrator
    template <typename First, typename B2, typename  B3 , template <typename T1, typename T2, typename T3> class Decorator >
    struct is_storage<Decorator<First, B2, B3>  *  > : public is_storage<typename First::basic_type*>
    {};

#endif

    //Decorator is the integrator
    template <typename BaseType , template <typename T, ushort_t O> class Decorator, ushort_t Order >
    struct is_storage<Decorator<BaseType, Order>  *  > : public is_storage<typename BaseType::basic_type*>
    {};

} // namespace gridtools
