#pragma once
#include "base_storage_impl.h"
#include "hybrid_pointer.h"
#include "../common/array.h"

/**@file
   @brief Implementation of the main storage class, used by all backends, for temporary and non-temporary storage

   We define here an important naming convention. We call:

   - the data fields: are contiguous chunks of memory, accessed by 3 (by default, but not necessarily) indexes.
   These structures are univocally defined by 3 (by default) integers. These are currently 2 strides and the total size of the chunks. Note that (in 3D) the relation between these quantities (\f$stride_1\f$, \f$stride_2\f$ and \f$size\f$) and the dimensions x, y and z can be (depending on the storage layout chosen)
   \f[
   size=x*y*z //
   stride_2=x*y //
   stride_1=x .
   \f]
   The quantities \f$size\f$, \f$stride_2\f$ and \f$stride_1\f$ are arranged respectively in m_strides[0], m_strides[1], m_strides[2].
   - the data snapshot: is a single pointer to one data field. The snapshots are arranged in the storages on a 1D array, regardless of the dimension and snapshot they refer to. The arg_type (or arg_decorator) class is
   responsible of computing the correct offests (relative to the given dimension) and address the storages correctly.
   - the storage: is an instance of any storage class, and can contain one or more fields and dimension. Every dimension consists of one or several snaphsots of the fields
   (e.g. if the time T is the current dimension, 3 snapshots can be the fields at t, t+1, t+2)

   The basic_storage class has a 1-1 relation with the data fields, while the subclasses extend the concept of storage to the structure represented in the ASCII picture below.

   NOTE: the constraint of the data fields accessed by the same storage class are the following:
   - the memory layout (strides, space dimensions) is one for all the data fields, all the snapshots
     share the same access index
\verbatim
############### 2D Storage ################
#                    ___________\         #
#                      time     /         #
#                  | |*|*|*|*|*|*|        #
# space, pressure  | |*|*|*|              #
#    energy,...    v |*|*|*|*|*|          #
#                                         #
#                     ^ ^ ^ ^ ^ ^         #
#                     | | | | | |         #
#                      snapshots          #
#                                         #
############### 2D Storage ################
\endverbatim

   The final storage which is effectly instantiated must be "clonable to the GPU", i.e. it must derive from the clonable_to_gpu struct.
   This is achieved by defining a class with multiple inheritance.

   NOTE CUDA: It is important when subclassing from a storage object to reimplement the __device__ copy constructor, and possibly the method 'copy_data_to_gpu' which are used when cloning the class to the CUDA device.
*/

namespace gridtools {

    /**
     * @brief Type to indicate that the type is not decided yet
     */
    template <typename RegularStorageType>
    struct no_storage_type_yet {
        typedef RegularStorageType type;
        typedef typename  type::layout layout;
        typedef typename  type::const_iterator_type const_iterator_type;
        typedef typename  RegularStorageType::basic_type basic_type;
        typedef typename RegularStorageType::original_storage original_storage;
        typedef typename RegularStorageType::pointer_type pointer_type;
        static const ushort_t n_width=basic_type::n_width;
        static const ushort_t filed_dimensions=basic_type::field_dimensions;
        typedef void storage_type;
        typedef typename RegularStorageType::iterator_type iterator_type;
        typedef typename RegularStorageType::value_type value_type;
        static const ushort_t space_dimensions=RegularStorageType::space_dimensions;
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
       @brief main class for the basic storage

       The base_storage class contains one snapshot. It univocally defines
       the access pattern with three integers: the total storage sizes and
       the two strides different from one.
    */
    template < typename PointerType,
               typename Layout,
               bool IsTemporary = false,
               short_t FieldDimension=1
               >
    struct base_storage
    {
        typedef base_storage<PointerType, Layout, IsTemporary, FieldDimension> type;
        typedef Layout layout;
        typedef PointerType pointer_type;
        typedef typename pointer_type::pointee_t value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;

        // TODO: Keep only one of these
        typedef base_storage<PointerType, Layout, IsTemporary, FieldDimension> basic_type;
        typedef base_storage<PointerType, Layout, IsTemporary, FieldDimension> original_storage;

        static const bool is_temporary = IsTemporary;
        static const ushort_t n_width = 1;
        static const ushort_t space_dimensions = layout::length;
        //field_dimensions is the total dimension of the storage
        static const short_t field_dimensions = FieldDimension;

    public:

        template <typename T, typename U, bool B>
        friend std::ostream& operator<<(std::ostream &, base_storage<T,U, B> const & );

        /**@brief the parallel storage calls the empty constructor to do lazy initialization*/
        base_storage() :
            is_set( false ),
            m_name("default_storage")
            {}

#if defined(CXX11_ENABLED) && !defined( __CUDACC__)

        /**
           @brief 3D storage constructor
           \tparam FloatType is the floating point type passed to the constructor for initialization. It is a template parameter in order to match float, double, etc...
        */
        template<typename FloatType=float_type, typename boost::enable_if<boost::is_float<FloatType>, int>::type=0>
        base_storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3, FloatType const& init=float_type(), char const* s="default storage") :
            is_set( true ),
            m_name(s),
            m_dims(),
            m_strides()
            {
                GRIDTOOLS_STATIC_ASSERT( boost::is_float<FloatType>::value, "The initialization value in the storage constructor must be a floating point number (e.g. 1.0). \nIf you want to store an integer you have to split construction and initialization \n(using the member \"initialize\"). This because otherwise the initialization value would be interpreted as an extra dimension")
                    setup(dim1, dim2, dim3);
                allocate();
                initialize(init, 1);
            }

        /**@BRIEF generic multidimensional constructor

           There are two possible types of storage dimension. One (space dimension) defines the number of indexes
           used to access a contiguous chunk of data. The other (field dimension) defines the number of pointers
           to the data chunks (i.e. the number of snapshots) contained in the storage. This constructor
           allows to create a storage with arbitrary space dimensions. The extra dimensions can be
           used e.g. to perform extra inner loops, besides the standard ones on i,j and k.

           The number of arguments must me equal to the space dimensions of the specific field (template parameter)
        */
        template <class ... UIntTypes, typename Dummy = typename boost::enable_if_c<accumulate(logical_and(),  boost::is_integral<UIntTypes>::type::value ... ), bool >::type >
        base_storage(  UIntTypes const& ... args  ) :
            is_set( false ),
            m_name("default_storage"),
            m_dims(),
            m_strides()
            {
                setup(args ...);
            }

        template<typename ... UInt>
        void setup(UInt const& ... dims)
            {
                assign<space_dimensions-1>::apply(m_dims, std::tie(dims...));
                BOOST_STATIC_ASSERT(sizeof...(UInt)==space_dimensions);
                BOOST_STATIC_ASSERT(field_dimensions>0);
                m_strides[0] = accumulate( multiplies(), dims...) ;
                _impl::assign_strides<(short_t)(space_dimensions-2), (short_t)(space_dimensions-1), layout>::apply(&m_strides[0], dims...);

#ifdef PEDANTIC
                //the following assert fails when we passed an argument to the arbitrary dimensional storage constructor which is not an unsigned integer (type uint_t).
                //You only have to pass the dimension sizes to this constructor, maybe you have to explicitly cast the value
                BOOST_STATIC_ASSERT(accumulate(logical_and(), sizeof(UInt) == sizeof(uint_t) ... ) );
#endif
            }

#else //CXX11_ENABLED

        /**@brief default constructor
           sets all the data members given the storage dimensions
        */
   base_storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3,
           value_type init = value_type(0.), char const* s="default storage" ) :
       is_set( true )
       , m_name(s)
            {
                setup(dim1, dim2, dim3);
                allocate();
                initialize(init, 1);
                set_name(s);
            }


        /**@brief default constructor
           sets all the data members given the storage dimensions
        */
   base_storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3,
           value_type (*lambda)(uint_t const&, uint_t const&, uint_t const&), char const* s="default storage" ):
       is_set( true )
       , m_name(s)
            {
                setup(dim1, dim2, dim3);
                allocate();
                initialize(lambda, 1);
                set_name(s);
            }

        void setup(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3)
            {
                m_dims[0]=dim1;
                m_dims[1]=dim2;
                m_dims[2]=dim3;

                m_strides[0]=( ((layout::template at_<0>::value < 0)?1:dim1) * ((layout::template at_<1>::value < 0)?1:dim2) * ((layout::template at_<2>::value < 0)?1:dim3) );
                m_strides[1]=( (m_strides[0]<=1)?0:layout::template find_val<2,short_t,1>(dim1,dim2,dim3)*layout::template find_val<1,short_t,1>(dim1,dim2,dim3) );
                m_strides[2]=( (m_strides[1]<=1)?0:layout::template find_val<2,short_t,1>(dim1,dim2,dim3) );
            }

#endif //CXX11_ENABLED

        /**@brief 3D constructor with the storage pointer provided externally

           This interface handles the case in which the storage is allocated from the python interface. Since this storege gets freed inside python, it must be instantiated as a
           'managed outside' wrap_pointer. In this way the storage destructor will not free the pointer.*/
        template<typename FloatType>
        explicit base_storage(uint_t const& dim1, uint_t const& dim2, uint_t const& dim3, FloatType* ptr, char const* s="default storage"
            ):
            is_set( true ),
            m_name(s)
            {
      m_fields[0]=pointer_type(ptr, true);
                setup(dim1, dim2, dim3);
                set_name(s);
            }

        /**@brief destructor: frees the pointers to the data fields which are not managed outside */
        virtual ~base_storage(){
            for(ushort_t i=0; i<field_dimensions; ++i)
                if(!m_fields[i].managed())
                    m_fields[i].free_it();
        }

        /**@brief device copy constructor*/
        template<typename T>
        __device__
        base_storage(T const& other)
            :
            is_set(other.is_set)
            , m_name(other.m_name)
            , m_fields(other.m_fields)
            {
                // for (uint_t i=0; i< field_dimensions; ++i)
                //     m_fields[i]=pointer_type(other.m_fields[i]);
                m_dims[0]=other.m_dims[0];
                m_dims[1]=other.m_dims[1];
                m_dims[2]=other.m_dims[2];
                m_strides[0] = other.size();
                m_strides[1] = other.strides(1);
                m_strides[2] = other.strides(2);
            }

        void allocate(ushort_t const& dims=FieldDimension){
            is_set=true;
            for(ushort_t i=0; i<dims; ++i)
                m_fields[i]=pointer_type(size());
        }

        /** @brief initializes with a constant value */
        GT_FUNCTION
        void initialize(value_type const& init, ushort_t const& dims=field_dimensions)
            {
#ifdef _GT_RANDOM_INPUT
                srand(12345);
#endif
                for(ushort_t f=0; f<dims; ++f)
                {
                    for (uint_t i = 0; i < size(); ++i)
                    {
#ifdef _GT_RANDOM_INPUT
                        (m_fields[f])[i] = init * rand();
#else
                        (m_fields[f])[i] = init;
#endif
                    }
                }
            }


        /** @brief initializes with a lambda function */
        GT_FUNCTION
        void initialize(value_type (*lambda)(uint_t const&, uint_t const&, uint_t const&), ushort_t const& dims=field_dimensions)
            {
                for(ushort_t f=0; f<dims; ++f)
                {
                    for (uint_t i=0; i<this->m_dims[0]; ++i)
                        for (uint_t j=0; j<this->m_dims[1]; ++j)
                            for (uint_t k=0; k<this->m_dims[2]; ++k)
                                (m_fields[f])[_index(strides(),i,j,k)]=lambda(i, j, k);
                }
            }

        /**@brief sets the name of the current field*/
        GT_FUNCTION
        void set_name(char const* const& string){
            m_name=string;
        }

        /** @brief copies the data field to the GPU */
        GT_FUNCTION_WARNING
        void copy_data_to_gpu() const {
            for (uint_t i=0; i<field_dimensions; ++i)
                m_fields[i].update_gpu();
        }

        static void text() {
            std::cout << BOOST_CURRENT_FUNCTION << std::endl;
        }

        /** @brief update the GPU pointer */
        void h2d_update(){
            for (uint_t i=0; i<field_dimensions; ++i)
                m_fields[i].update_gpu();
        }

        /** @brief updates the CPU pointer */
        void d2h_update(){
            for (uint_t i=0; i<field_dimensions; ++i)
                m_fields[i].update_cpu();
        }

        /** @brief prints debugging information */
        void info() const {
            std::cout << dims<0>() << "x"
                      << dims<1>() << "x"
                      << dims<2>() << ", "
                      << std::endl;
        }

        /** @brief returns the first memory addres of the data field */
        GT_FUNCTION
        const_iterator_type min_addr() const {
            return &((m_fields[0])[0]);
        }


        /** @brief returns the last memry address of the data field */
        GT_FUNCTION
        const_iterator_type max_addr() const {
            return &((m_fields[0])[/*m_size*/m_strides[0]]);
        }

#ifdef CXX11_ENABLED
        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k) */
        template <typename ... UInt>
        GT_FUNCTION
        value_type& operator()(UInt const& ... dims) {
#ifndef __CUDACC__
            assert(_index(strides(),dims...) < size());
#endif
            return (m_fields[0])[_index(strides(),dims...)];
        }


        /** @brief returns (by const reference) the value of the data field at the coordinates (i, j, k) */
        template <typename ... UInt>
        GT_FUNCTION
        value_type const & operator()(UInt const& ... dims) const {
#ifndef __CUDACC__
            assert(_index(strides(),dims...) < size());
#endif
            return (m_fields[0])[_index(strides(),dims...)];
        }
#else //CXX11_ENABLED

        /** @brief returns (by reference) the value of the data field at the coordinates (i, j, k) */
        GT_FUNCTION
        value_type& operator()(uint_t const& i, uint_t const& j, uint_t const& k) {
#ifndef __CUDACC__
            assert(_index(strides(),i,j,k) < size());
#endif
            return (m_fields[0])[_index(strides(),i,j,k)];
        }


        /** @brief returns (by const reference) the value of the data field at the coordinates (i, j, k) */
        GT_FUNCTION
        value_type const & operator()(uint_t const& i, uint_t const& j, uint_t const& k) const {
#ifndef __CUDACC__
            assert(_index(strides(),i,j,k) < size());
#endif
            return (m_fields[0])[_index(strides(),i,j,k)];
        }
#endif
        /**@brief returns the size of the data field*/
        GT_FUNCTION
        uint_t const& size() const {
            return m_strides[0];
        }

        /**@brief prints the first values of the field to standard output*/
        void print() const {
            print(std::cout);
        }

        /**@brief prints a single value of the data field given the coordinates*/
        void print_value(uint_t i, uint_t j, uint_t k){ printf("value(%d, %d, %d)=%f, at index %d on the data\n", i, j, k, (m_fields[0])[_index(i, j, k)], _index(i, j, k));}

        static const std::string info_string;


        /**@brief return the stride for a specific coordinate, given the vector of strides
           Coordinates 0,1,2 correspond to i,j,k respectively*/
        template<uint_t Coordinate, typename StridesVector>
        GT_FUNCTION
        static constexpr uint_t strides(StridesVector const& RESTRICT strides_){
            return ((vec_max<typename layout::layout_vector_t>::value < 0) ? 0:(( layout::template at_<Coordinate>::value == vec_max<typename layout::layout_vector_t>::value ) ? 1 : ((strides_[layout::template at_<Coordinate>::value/*+1*/]))));//POL TODO explain the fact that here there was a +1
        }

        /**@brief printing a portion of the content of the data field*/
        template <typename Stream>
        void print(Stream & stream, uint_t t=0) const {
            stream << " (" << m_strides[1] << "x"
                   << m_strides[2] << "x"
                   << 1 << ")"
                   << std::endl;
            stream << "| j" << std::endl;
            stream << "| j" << std::endl;
            stream << "v j" << std::endl;
            stream << "---> k" << std::endl;

            ushort_t MI=12;
            ushort_t MJ=12;
            ushort_t MK=12;
            for (uint_t i = 0; i < dims<0>(); i += std::max(( uint_t)1,dims<0>()/MI)) {
                for (uint_t j = 0; j < dims<1>(); j += std::max(( uint_t)1,dims<1>()/MJ)) {
                    for (uint_t k = 0; k < dims<2>(); k += std::max(( uint_t)1,dims<1>()/MK))
                    {
                        stream << "["
                            // << i << ","
                            // << j << ","
                            // << k << ")"
                               <<  (m_fields[t])[_index(strides(), i,j,k)] << "] ";
                    }
                    stream << std::endl;
                }
                stream << std::endl;
            }
            stream << std::endl;
        }

        /**@brief returning the index of the memory address corresponding to the specified (i,j,k) coordinates.
           This method depends on the strategy used (either naive or blocking). In case of blocking strategy the
           index for temporary storages is computed in the subclass gridtools::host_tmp_storge
           NOTE: this version will be preferred over the templated overloads,
           NOTE: the strides are passed in as an argument, and the result can be a constexpr also when the storage is not.
        */
        template<typename StridesVector>
        GT_FUNCTION
#ifndef __CUDACC__
        constexpr
#endif
        uint_t  _index(StridesVector const& RESTRICT strides_, uint_t const& i, uint_t const& j, uint_t const&  k) const {
            return strides_[0]
                * layout::template find_val<0,uint_t,0>(i,j,k) +
                strides_[1] * layout::template find_val<1,uint_t,0>(i,j,k) +
                layout::template find_val<2,uint_t,0>(i,j,k);
        }

        /**@brief straightforward interface*/
        GT_FUNCTION
        uint_t _index(uint_t const& i, uint_t const& j, uint_t const&  k) const { return _index(strides(), i, j, k);}

#ifdef CXX11_ENABLED
        /**
           @brief computing index to access the storage relative to the coordinates passed as parameters.

           This interface must be used with unsigned integers of type uint_t, and the result must be a positive integer as well
        */
        template <typename StridesVector, typename ... UInt>
        GT_FUNCTION
        constexpr
        uint_t _index(StridesVector const& RESTRICT strides_, UInt const& ... dims) const {
#ifndef __CUDACC__
            typedef boost::mpl::vector<UInt...> tlist;
            //boost::is_same<boost::mpl::_1, uint_t>
            typedef typename boost::mpl::find_if<tlist, boost::mpl::not_< boost::is_integral<boost::mpl::_1> > >::type iter;
            GRIDTOOLS_STATIC_ASSERT(iter::pos::value==sizeof...(UInt), "you have to pass in arguments of uint_t type")
#endif
            return _impl::compute_offset<space_dimensions, layout>::apply(strides_, dims ...);
        }
#endif

        /**
           @brief computing index to access the storage relative to the coordinates passed as parameters.

           This method returns signed integers of type int_t (used e.g. in iterate_domain)
        */
        template <typename OffsetTuple, typename StridesVector>
        GT_FUNCTION
        int_t _index(StridesVector const& RESTRICT strides_, OffsetTuple  const& tuple) const {
            return _impl::compute_offset<space_dimensions, layout>::apply(strides_, tuple);
        }


        /** @brief returns the memory access index of the element with coordinate (i,j,k)
            note: returns a signed int because it might be negative (used e.g. in iterate_domain)*/
        template<typename IntType, typename StridesVector>
        GT_FUNCTION
        int_t _index( StridesVector const& RESTRICT strides_, IntType* RESTRICT indices) const {

            return  _impl::compute_offset<space_dimensions, layout>::apply(strides_, indices);
        }

        /** @brief method to increment the memory address index by moving forward one step in the given Coordinate direction

            SFINAE: design pattern used to avoid the compilation of the overloaded method which are not used (and which would produce a compilation error)
        */
        template <uint_t Coordinate, enumtype::execution Execution, typename StridesVector>
        GT_FUNCTION
        void increment( uint_t* RESTRICT index_, StridesVector const& RESTRICT strides_/*, typename boost::enable_if_c< (layout::template pos_<Coordinate>::value >= 0) >::type* dummy=0*/){
            BOOST_STATIC_ASSERT(Coordinate < space_dimensions);
            if(layout::template at_< Coordinate >::value >=0)//static if
            {
                increment_policy<Execution>::apply(*index_ , strides<Coordinate>(strides_));
            }
        }

        /** @brief method to increment the memory address index by moving forward a given number of step in the given Coordinate direction
            \tparam Coordinate: the dimension which is being incremented (0=i, 1=j, 2=k, ...)
            \param steps: the number of steps of the increment
            \param index: the output index being set
        */
        template <uint_t Coordinate, enumtype::execution Execution, typename StridesVector>
        GT_FUNCTION
        void increment(uint_t const& steps_, uint_t* RESTRICT index_, StridesVector const& RESTRICT strides_){
            BOOST_STATIC_ASSERT(Coordinate < space_dimensions);
            if( layout::template at_< Coordinate >::value >= 0 )//static if
            {
                increment_policy<Execution>::apply(*index_ , strides<Coordinate>(strides_)*steps_);
            }
        }

        GT_FUNCTION
        void set_index(uint_t value, uint_t* index){
            *index=value;
        }

        template <uint_t Coordinate, typename StridesVector >
        GT_FUNCTION
        void initialize(uint_t const& steps_, uint_t const& /*block*/, StridesVector const& RESTRICT strides_, uint_t* RESTRICT index_){
            BOOST_STATIC_ASSERT(Coordinate < space_dimensions);
            if( layout::template at_< Coordinate >::value >= 0 )//static if
            {
                *index_+=strides<Coordinate>(strides_)*steps_;
            }
        }

        /**@brief returns the data field*/
        GT_FUNCTION
        pointer_type const& data() const {return (m_fields[0]);}

        /** @brief returns a const pointer to the data field*/
        GT_FUNCTION
        pointer_type const* fields() const {return &(m_fields[0]);}

        /** @brief returns the dimension fo the field along I*/
        template<ushort_t I>
        GT_FUNCTION
        uint_t dims() const {return m_dims[I];}

        /** @brief returns the dimension fo the field along I*/
        GT_FUNCTION
        uint_t dims(const ushort_t I) const {return m_dims[I];}

        /**@brief returns the storage strides
         */
        GT_FUNCTION
        uint_t const& strides(ushort_t i) const {
            // m_strides[0] contains the whole storage dimension."
            assert(i!=0);
            return m_strides[i];
        }

        /**@brief returns the storage strides
         */
        GT_FUNCTION
        uint_t const* strides() const {
            GRIDTOOLS_STATIC_ASSERT(space_dimensions>1, "one dimensional storage")
            return (&m_strides[1]);
        }

    protected:
        bool is_set;
        const char* m_name;
#ifdef __CUDACC__ // this is related to the fact that the gridtools::array should not be templated to a const type when CXX11 disabled
        pointer_type m_fields[field_dimensions];
#else
        array<pointer_type, field_dimensions> m_fields;
#endif
        array<uint_t, space_dimensions> m_dims;
        array<uint_t, space_dimensions> m_strides;

    };

template < typename PointerType,
               typename Layout,
               bool IsTemporary,
               short_t FieldDimension>
const short_t base_storage<PointerType, Layout, IsTemporary, FieldDimension>::field_dimensions;

    /** @brief storage class containing a buffer of data snapshots


    */
    template < typename Storage, short_t ExtraWidth>
    struct storage_list : public Storage
    {

        typedef storage_list<Storage, ExtraWidth> type;
   /*If the following assertion fails, you probably set one field dimension to contain zero (or negative) snapshots. Each field dimension must contain one or more snapshots.*/
   GRIDTOOLS_STATIC_ASSERT(ExtraWidth>0, "you probably set one field dimension to contain zero (or negative) snapshots. Each field dimension must contain one or more snapshots.")
        typedef Storage super;
        typedef typename super::pointer_type pointer_type;

        typedef typename super::original_storage original_storage;
        typedef typename super::iterator_type iterator_type;
        typedef typename super::value_type value_type;

        //default constructor
        storage_list(): super(){}

#ifdef CXX11_ENABLED
        /**@brief default constructor*/
        template<typename ... UIntTypes>
        explicit storage_list(UIntTypes const& ... args ): super( args ... ) {
        }
#else
        /**@brief default constructor*/
        explicit storage_list(uint_t const& d1, uint_t const& d2, uint_t const& d3 ): super( d1, d2, d3 ) {
        }
#endif


        /**@brief destructor: frees the pointers to the data fields */
        virtual ~storage_list(){
        }

        /**@brief device copy constructor*/
        template <typename T>
        __device__
        storage_list(T const& other)
            : super(other)
            {
                //GRIDTOOLS_STATIC_ASSERT(n_width==T::n_width, "Dimension analysis error: copying two vectors with different dimensions");
            }

        /**@brief copy all the data fields to the GPU*/
        GT_FUNCTION_WARNING
        void copy_data_to_gpu(){
            //the fields are otherwise not copied to the gpu, since they are not inserted in the storage_pointers fusion vector
            for (uint_t i=0; i< super::field_dimensions; ++i)
                super::m_fields[i].update_gpu();
        }

        using super::setup;

        /**
            @brief returns a const reference to the specified data snapshot

            \param index the index of the snapshot in the array
        */
        GT_FUNCTION
        pointer_type const& get_field(int index) const {return super::m_fields[index];};

        /**@brief swaps the argument with the last data snapshots*/
        GT_FUNCTION
        void swap(pointer_type & field){
            //the integration takes ownership over all the pointers?
            //cycle in a ring
            pointer_type swap(super::m_fields[super::field_dimensions-1]);
            super::m_fields[super::field_dimensions-1]=field;
            field = swap;
        }

        /**@brief adds a given data field at the front of the buffer
           \param field the pointer to the input data field
           NOTE: better to shift all the pointers in the array, because we do this seldomly, so that we don't need to keep another indirection when accessing the storage ("stateless" buffer)
        */
        GT_FUNCTION
        void push_front( pointer_type& field, uint_t const& from=(uint_t)0, uint_t const& to=(uint_t)(n_width)){
            //cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when accessing the storage (stateless buffer)
            for(uint_t i=from+1;i<to;i++) super::m_fields[i]=super::m_fields[i-1];
            super::m_fields[from]=(field);
        }

        //the time integration takes ownership over all the pointers?
        /**TODO code repetition*/
        GT_FUNCTION
        void advance(uint_t from=(uint_t)0, uint_t to=(uint_t)(n_width)){
            pointer_type tmp(super::m_fields[to-1]);
            for(uint_t i=from+1;i<to;i++) super::m_fields[i]=super::m_fields[i-1];
            super::m_fields[from]=tmp;
        }

        /**@brief printing the first values of all the snapshots contained in the discrete field*/
        void print() {
            print(std::cout);
        }

        /**@brief printing the first values of all the snapshots contained in the discrete field, given the output stream*/
        template <typename Stream>
        void print(Stream & stream) {
            for (ushort_t t=0; t<super::field_dimensions; ++t)
            {
                stream<<" Component: "<< t+1<<std::endl;
                original_storage::print(stream, t);
            }
        }

        static const ushort_t n_width = ExtraWidth+1;

    };

    /**@brief specialization: if the width extension is 0 we fall back on the base storage*/
    template < typename Storage>
    struct storage_list<Storage, 0> : public Storage
    {
        typedef typename Storage::basic_type basic_type;
        typedef Storage super;
        typedef typename Storage::original_storage original_storage;

        //default constructor
        storage_list(): super(){}

#ifdef CXX11_ENABLED
        /**@brief default constructor*/
        template<typename ... UIntTypes>
        explicit storage_list(UIntTypes const& ... args ): Storage( args ... ) {
        }
#else
        /**@brief default constructor*/
        explicit storage_list(uint_t const& d1, uint_t const& d2, uint_t const& d3 ): Storage( d1, d2, d3 ) {
        }
#endif

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~storage_list(){
        }

        using super::setup;

   /**dimension number of snaphsots for the current field dimension*/
        static const ushort_t n_width = Storage::n_width;

        /**@brief device copy constructor*/
        template<typename T>
        __device__
        storage_list(T const& other)
            : Storage(other)
            {}
    };

#if defined(CXX11_ENABLED) && !defined(__CUDACC__)
    /** @brief traits class defining some useful compile-time counters
     */
    template < typename First, typename  ...  StorageExtended>
    struct dimension_extension_traits
    {
        //total number of snapshots in the discretized field
        static const ushort_t n_fields=First::n_width + dimension_extension_traits<StorageExtended  ...  >::n_fields ;
        //the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
        static const short_t n_width=First::n_width;
        //the number of dimensions (i.e. the number of different fields)
        static const ushort_t n_dimensions=  dimension_extension_traits<StorageExtended  ...  >::n_dimensions  +1 ;
        //the current field extension
        //n_fields-1 because the storage_list takes the EXTRA width as argument, not the total width.
        typedef storage_list<First, n_fields-1>  type;
        // typedef First type;
        typedef dimension_extension_traits<StorageExtended ... > super;
    };


    template<typename T>
    struct get_fields{
        using type = static_int<T::n_fields>;
    };

    template<typename T>
    struct get_value{
        using type = static_int<T::value>;
    };

    template<typename Storage, uint_t Id, uint_t IdMax>
    struct compute_storage_offset{

        GRIDTOOLS_STATIC_ASSERT(IdMax>=Id && Id>=0, "Library internal error")
            typedef typename boost::mpl::eval_if_c<IdMax-Id==0, get_fields<typename Storage::super> , get_value<compute_storage_offset<typename Storage::super, Id+1, IdMax> > >::type type;
        static const uint_t value=type::value;
    };

#endif

    /**@brief fallback in case the snapshot we try to access exceeds the width diemnsion assigned to a discrete scalar field*/
    struct dimension_extension_null{
        static const ushort_t n_fields=0;
        static const short_t n_width=0;
        static const ushort_t n_dimensions=0;
        typedef struct error_index_too_large1{} type;
        typedef struct error_index_too_large2{} super;
    };

/**@brief template specialization at the end of the recustion.*/
    template < typename First>
    struct dimension_extension_traits
#if defined(CXX11_ENABLED) && !defined(__CUDACC__)
    <First>
#endif
    {
        static const ushort_t n_fields=First::n_width;
        static const short_t n_width=First::n_width;
        static const ushort_t n_dimensions= 1 ;
        typedef First type;
        typedef dimension_extension_null super;
    };

#if !defined(CXX11_ENABLED) || defined(__CUDACC__)// big code repetition

    /** @brief non-C++11 crap
     */
    template < typename First, typename Second>
    struct dimension_extension_traits2// : public dimension_extension_traits<StorageExtended ... >
    {
        //total number of snapshots in the discretized field
        static const ushort_t n_fields=First::n_width + dimension_extension_traits< Second >::n_fields ;
        //the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
        static const short_t n_width=First::n_width;
        //the number of dimensions (i.e. the number of different fields)
        static const ushort_t n_dimensions=  dimension_extension_traits< Second>::n_dimensions  +1 ;
        //the current field extension
   //n_fields-1 because the storage_list takes the EXTRA width as argument, not the total width.
   typedef storage_list<First, n_fields-1>  type;
   // typedef First type;
   typedef dimension_extension_traits< Second> super;
    };

    template < typename First, typename Second, typename Third>
    struct dimension_extension_traits3// : public dimension_extension_traits<StorageExtended ... >
    {
        //total number of snapshots in the discretized field
        static const ushort_t n_fields=First::n_width + dimension_extension_traits2< Second, Third >::n_fields ;
        //the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
        static const short_t n_width=First::n_width;
        //the number of dimensions (i.e. the number of different fields)
        static const ushort_t n_dimensions=  dimension_extension_traits2< Second, Third >::n_dimensions  +1 ;
        //the current field extension
   //n_fields-1 because the storage_list takes the EXTRA width as argument, not the total width.
   typedef storage_list<First, n_fields-1>  type;
   // typedef First type;
   typedef dimension_extension_traits2< Second, Third > super;
    };
#endif //CXX11_ENABLED


#if defined( CXX11_ENABLED ) && !defined( __CUDACC__ )
    /**@brief implements the discretized field structure

       It is a collection of arbitrary length discretized scalar fields.
    */
    template <typename First,  typename  ...  StorageExtended>
    struct data_field : public dimension_extension_traits<First, StorageExtended ... >::type/*, clonable_to_gpu<data_field<First, StorageExtended ... > >*/
    {
        typedef data_field<First, StorageExtended...> type;
        typedef typename dimension_extension_traits<First, StorageExtended ... >::type super;
        typedef dimension_extension_traits<First, StorageExtended ...  > traits;
        typedef typename super::pointer_type pointer_type;
        typedef typename  super::basic_type basic_type;
        typedef typename super::original_storage original_storage;
        static const short_t n_width=sizeof...(StorageExtended)+1;

        /**@brief constructor given the space boundaries*/
        template<typename ... UIntTypes>
        data_field(  UIntTypes const& ... args )
            : super(args...)
            {
       }
#else

        template <typename First, typename Second, typename Third>
        struct data_field : public dimension_extension_traits3<First, Second, Third >::type/*, clonable_to_gpu<data_field<First, StorageExtended ... > >*/
        {
            typedef data_field<First, Second, Third> type;
            typedef typename dimension_extension_traits3<First, Second, Third >::type super;
            typedef dimension_extension_traits3<First, Second, Third > traits;
            typedef typename super::pointer_type pointer_type;
            typedef typename  super::basic_type basic_type;
            typedef typename super::original_storage original_storage;
            static const short_t n_width=2+1;

            /**@brief constructor given the space boundaries*/
            data_field(  uint_t const& d1, uint_t const& d2, uint_t const& d3 )
                : super(d1, d2, d3)
            {
       }

#endif

   /**@brief default constructor*/
        data_field(): super(){}

   /**@brief device copy constructor*/
        template <typename T>
        __device__
        data_field( T const& other )
            : super(other)
            {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~data_field(){
        }

        using super::setup;

        /**@brief pushes a given data field at the front of the buffer for a specific dimension
           \param field the pointer to the input data field
           \tparam dimension specifies which field dimension we want to access
        */
        template<uint_t dimension
#ifdef CXX11_ENABLED
=1
#endif
>
        GT_FUNCTION
        void push_front( pointer_type& field ){//copy constructor
            //cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when accessing the storage (stateless storage)

            /*If the following assertion fails your field dimension is smaller than the dimension you are trying to access*/
            BOOST_STATIC_ASSERT(n_width>dimension);
            /*If the following assertion fails you specified a dimension which does not contain any snapshot. Each dimension must contain at least one snapshot.*/
            BOOST_STATIC_ASSERT(n_width<=traits::n_fields);
            uint_t const indexFrom=_impl::access<n_width-dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<n_width-dimension-1, traits>::type::n_fields;
            super::push_front(field, indexFrom, indexTo);
        }

   /**@brief Pushes the given storage as the first snapshot at the specified field dimension*/
        template<uint_t dimension
#ifdef CXX11_ENABLED
=1
#endif
>
        GT_FUNCTION
        void push_front( pointer_type& field, typename super::value_type const& value ){//copy constructor
       for (uint_t i=0; i<super::size(); ++i)
           field[i]=value;
       push_front<dimension>(field);
   }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim, short_t snapshot>
#endif
   void set( pointer_type& field)
       {
           super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot]=field;
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input constant value

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param val the initializer value
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim, short_t snapshot>
#endif
   void set(/* pointer_type& field,*/ typename super::value_type const& val)
       {
           for (uint_t i=0; i<super::size(); ++i)
               (super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot])[i]=val;
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input lambda function
      TODO: this should be merged with the boundary conditions code (repetition)

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param lambda the initializer function
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim, short_t snapshot  >
#endif
   void set( typename super::value_type (*lambda)(uint_t const&, uint_t const&, uint_t const&))
       {
      for (uint_t i=0; i<this->m_dims[0]; ++i)
          for (uint_t j=0; j<this->m_dims[1]; ++j)
         for (uint_t k=0; k<this->m_dims[2]; ++k)
             for (uint_t i=0; i<super::size(); ++i)
               (super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot])
                   [super::_index(super::strides(), i,j,k)]=
                   lambda(i, j, k);
       }


   /**@biref gets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim  , short_t snapshot  >
#endif
   pointer_type& get( )
       {
      return super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot];
       }


        /**
           @brief returns the index (in the array of data snapshots) corresponding to the specified offset
           basically it returns offset unless it is negative or it exceeds the size of the internal array of snapshots. In the latter case it returns offset modulo the size of the array.
           In the former case it returns the array size's complement of -offset.
        */
        GT_FUNCTION
        static constexpr ushort_t get_index (short_t const& offset) {
            return (offset+n_width)%n_width;
        }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension, at the specified coordinates

           If on the device, it calls the API to set the memory on the device
      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param value the value to be set
        */
#ifdef CXX11_ENABLED
            template<typename StridesVector, short_t field_dim=0, short_t snapshot=0>
#else
            template<typename StridesVector, short_t field_dim  , short_t snapshot  >
#endif
            void set_value( StridesVector const& strides_, typename super::value_type const& value, uint_t const& x, uint_t const& y, uint_t const& z)
                {
                    super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot].set(value, super::_index(strides_,x, y, z));
                }


        /**@biref gets a given value as the given field i,j,k coordinates

           \tparam field_dim the given field dimenisons
           \tparam snapshot the snapshot (relative to the dimension field_dim) to be acessed
           \param i index in the horizontal direction
           \param j index in the horizontal direction
           \param k index in the vertical direction
        */
#ifdef CXX11_ENABLED
   template<short_t field_dim=0, short_t snapshot=0>
#else
   template<short_t field_dim, short_t snapshot  >
#endif
        typename super::value_type& get_value( uint_t const& i, uint_t const& j, uint_t const& k )
      {
                    return get<field_dim, snapshot>()[super::_index(super::strides(),i,j,k)];
      }

        /**@biref ODE advancing for a single dimension

           it advances the supposed finite difference scheme of one step for a specific field dimension
           \tparam dimension the dimension to be advanced
           \param offset the number of steps to advance
        */
        template<uint_t dimension
#ifdef CXX11_ENABLED
=1
#endif
>
        GT_FUNCTION
        void advance(){
            BOOST_STATIC_ASSERT(dimension<traits::n_dimensions);
            uint_t const indexFrom=_impl::access<dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<dimension-1, traits>::type::n_fields;

            super::advance(indexFrom, indexTo);
        }

        /**@biref ODE advancing for all dimension

           shifts the rings of solutions of one position,
           it advances the finite difference scheme of one step for all field dimensions.
        */
        GT_FUNCTION
        void advance_all(){
            _impl::advance_recursive<n_width>::apply(const_cast<data_field*>(this));
        }

    };

#if !defined( CXX11_ENABLED ) || defined ( __CUDACC__ )
        template <typename First>
        struct data_field1 : public dimension_extension_traits<First >::type/*, clonable_to_gpu<data_field<First, StorageExtended ... > >*/
        {
            typedef data_field1<First> type;
            typedef typename dimension_extension_traits<First >::type super;
            typedef dimension_extension_traits<First > traits;
            typedef typename super::pointer_type pointer_type;
            typedef typename  super::basic_type basic_type;
            typedef typename super::original_storage original_storage;
            static const short_t n_width=2+1;

            /**@brief constructor given the space boundaries*/
            data_field1(  uint_t const& d1, uint_t const& d2, uint_t const& d3 )
                : super(d1, d2, d3)
            {
       }

   /**@brief device copy constructor*/
        template <typename T>
        __device__
        data_field1( T const& other )
            : super(other)
            {}

        /**@brief destructor: frees the pointers to the data fields */
        virtual ~data_field1(){
   }

        /**@brief pushes a given data field at the front of the buffer for a specific dimension
           \param field the pointer to the input data field
      \tparam dimension specifies which field dimension we want to access
        */
        template<uint_t dimension>
        GT_FUNCTION
        void push_front( pointer_type& field ){//copy constructor
            //cycle in a ring: better to shift all the pointers, so that we don't need to keep another indirection when accessing the storage (stateless storage)

       /*If the following assertion fails your field dimension is smaller than the dimension you are trying to access*/
            BOOST_STATIC_ASSERT(n_width>dimension);
       /*If the following assertion fails you specified a dimension which does not contain any snapshot. Each dimension must contain at least one snapshot.*/
            BOOST_STATIC_ASSERT(n_width<=traits::n_fields);
            uint_t const indexFrom=_impl::access<n_width-dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<n_width-dimension-1, traits>::type::n_fields;
       super::push_front(field, indexFrom, indexTo);
        }

   /**@brief Pushes the given storage as the first snapshot at the specified field dimension*/
        template<uint_t dimension>
        GT_FUNCTION
        void push_front( pointer_type& field, typename super::value_type const& value ){//copy constructor
       for (uint_t i=0; i<super::size(); ++i)
           field[i]=value;
       push_front<dimension>(field);
   }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
   template<short_t field_dim, short_t snapshot>
   void set( pointer_type& field)
       {
      super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot]=field;
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input constant value

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param val the initializer value
        */
   template<short_t field_dim, short_t snapshot>
   void set( pointer_type& field, typename super::value_type const& val)
       {
      for (uint_t i=0; i<super::size(); ++i)
          field[i]=val;
      set<field_dim, snapshot>(field);
       }

   /**@biref sets the given storage as the nth snapshot of a specific field dimension and initialize the storage with an input lambda function
      TODO: this should be merged with the boundary conditions code (repetition)

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
      \param lambda the initializer function
        */
   template<short_t field_dim, short_t snapshot  >
   void set( pointer_type& field, typename super::value_type (*lambda)(uint_t const&, uint_t const&, uint_t const&))
       {
      for (uint_t i=0; i<this->m_dims[0]; ++i)
          for (uint_t j=0; j<this->m_dims[1]; ++j)
         for (uint_t k=0; k<this->m_dims[2]; ++k)
             (field)[super::_index(super::strides(), i,j,k)]=lambda(i, j, k);
      set<field_dim, snapshot>(field);
       }


   /**@biref gets the given storage as the nth snapshot of a specific field dimension

      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param field the input storage
        */
   template<short_t field_dim  , short_t snapshot  >
   pointer_type& get( )
       {
      return super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot];
       }


   /**@biref sets the given storage as the nth snapshot of a specific field dimension, at the specified coordinates

           If on the device, it calls the API to set the memory on the device
      \tparam field_dim the given field dimenisons
      \tparam snapshot the snapshot of dimension field_dim to be set
      \param value the value to be set
        */
            template<typename StridesVector, short_t field_dim  , short_t snapshot  >
   void set_value( StridesVector const& strides_, typename super::value_type const& value, uint_t const& x, uint_t const& y, uint_t const& z)
                {
                    super::m_fields[_impl::access<n_width-(field_dim), traits>::type::n_fields + snapshot].set(value, super::_index(strides_,x, y, z));
                }


        /**@biref gets a given value as the given field i,j,k coordinates

           \tparam field_dim the given field dimenisons
           \tparam snapshot the snapshot (relative to the dimension field_dim) to be acessed
           \param i index in the horizontal direction
           \param j index in the horizontal direction
           \param k index in the vertical direction
        */
   template<short_t field_dim, short_t snapshot  >
        typename super::value_type& get_value( uint_t const& i, uint_t const& j, uint_t const& k )
      {
                    return get<field_dim, snapshot>()[super::_index(super::strides(),i,j,k)];
      }

   /**@biref ODE advancing for a single dimension

      it advances the supposed finite difference scheme of one step for a specific field dimension
      \tparam dimension the dimension to be advanced
      \param offset the number of steps to advance
        */
        template<uint_t dimension>
        GT_FUNCTION
        void advance(){
            BOOST_STATIC_ASSERT(dimension<traits::n_dimensions);
            uint_t const indexFrom=_impl::access<dimension, traits>::type::n_fields;
            uint_t const indexTo=_impl::access<dimension-1, traits>::type::n_fields;

            super::advance(indexFrom, indexTo);
        }

   /**@biref ODE advancing for all dimension

      shifts the rings of solutions of one position,
      it advances the finite difference scheme of one step for all field dimensions.
        */
        GT_FUNCTION
        void advance_all(){
       _impl::advance_recursive<n_width>::apply(const_cast<data_field1*>(this));
        }

#ifdef NDEBUG
    private:
        //for stdcout purposes
   data_field1();
#else
        data_field1(){}
#endif
    };
#endif
/** \addtogroup specializations Specializations
    Partial specializations
    @{
*/
    template <typename PointerType, typename Layout, bool IsTemporary, short_t Dim
               >
    const std::string base_storage<PointerType, Layout, IsTemporary, Dim
                                   >::info_string=boost::lexical_cast<std::string>("-1");

    template <typename PointerType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<PointerType,Y,false, Dim>*& >
        : boost::false_type
    {};

    template <typename PointerType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<PointerType,Y,true, Dim>*& >
        : boost::true_type
    {};

    template <typename PointerType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<PointerType,Y,false, Dim>* >
        : boost::false_type
    {};

    template <typename PointerType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<PointerType,Y,true, Dim>* >
        : boost::true_type
    {};

    template <typename PointerType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<PointerType,Y,false, Dim> >
        : boost::false_type
    {};

    template <typename PointerType, typename Y, short_t Dim>
    struct is_temporary_storage<base_storage<PointerType,Y,true, Dim> >
        : boost::true_type
    {};

    template <  template <typename T> class  Decorator, typename BaseType>
    struct is_temporary_storage<Decorator< BaseType > > : is_temporary_storage< typename BaseType::basic_type >
    {};
    template <  template <typename T> class Decorator, typename BaseType>
    struct is_temporary_storage<Decorator< BaseType >* > : is_temporary_storage< typename BaseType::basic_type* >
    {};
    template <  template <typename T> class Decorator, typename BaseType>
    struct is_temporary_storage<Decorator< BaseType >& > : is_temporary_storage< typename BaseType::basic_type& >
    {};
    template <  template <typename T> class Decorator, typename BaseType>
    struct is_temporary_storage<Decorator< BaseType >*& > : is_temporary_storage< typename BaseType::basic_type*& >
    {};

#ifdef CXX11_ENABLED
    //Decorator is the field
    template <template <typename ... T> class Decorator, typename First, typename ... BaseType>
    struct is_temporary_storage<Decorator<First, BaseType...> > : is_temporary_storage< typename First::basic_type >
    {};

    //Decorator is the field
    template <template <typename ... T> class Decorator, typename First, typename ... BaseType>
    struct is_temporary_storage<Decorator<First, BaseType...>* > : is_temporary_storage< typename First::basic_type* >
    {};

    //Decorator is the field
    template <template <typename ... T> class Decorator, typename First, typename ... BaseType>
    struct is_temporary_storage<Decorator<First, BaseType...>& > : is_temporary_storage< typename First::basic_type& >
    {};

    //Decorator is the field
    template <template <typename ... T> class Decorator, typename First, typename ... BaseType>
    struct is_temporary_storage<Decorator<First, BaseType...>*& > : is_temporary_storage< typename First::basic_type*& >
    {};
#else
        //Decorator is the field
        template <template <typename T1, typename T2, typename T3> class Decorator, typename First, typename B2, typename B3>
        struct is_temporary_storage<Decorator<First, B2, B3> > : is_temporary_storage< typename First::basic_type >
        {};

        //Decorator is the field
        template <template <typename T1, typename T2, typename T3> class Decorator, typename First, typename B2, typename B3>
        struct is_temporary_storage<Decorator<First, B2, B3>* > : is_temporary_storage< typename First::basic_type* >
        {};

        //Decorator is the field
        template <template <typename T1, typename T2, typename T3> class Decorator, typename First, typename B2, typename B3>
        struct is_temporary_storage<Decorator<First, B2, B3>& > : is_temporary_storage< typename First::basic_type& >
        {};

        //Decorator is the field
        template <template <typename T1, typename T2, typename T3> class Decorator, typename First, typename B2, typename B3>
        struct is_temporary_storage<Decorator<First, B2, B3>*& > : is_temporary_storage< typename First::basic_type*& >
        {};

#endif //CXX11_ENABLED
/**@}*/
    template <typename T, typename U, bool B>
    std::ostream& operator<<(std::ostream &s, base_storage<T,U, B> const & x ) {
        s << "base_storage <T,U," << " " << std::boolalpha << B << "> ";
        s << x.m_dims[0] << ", "
          << x.m_dims[1] << ", "
          << x.m_dims[2] << ". ";
        return s;
    }

#if defined(CXX11_ENABLED) && !defined( __CUDACC__ )
    template <typename F, typename ... T>
    std::ostream& operator<<(std::ostream &s, data_field< F, T... > const &) {
        return s << "field storage" ;
    }
#endif

} //namespace gridtools
