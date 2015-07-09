#pragma once
#include "base_storage_impl.hpp"
#include "hybrid_pointer.hpp"
#include "../common/array.hpp"

/**@file
   @brief Implementation of the \ref gridtools::base_storage "main storage class", used by all backends, for temporary and non-temporary storage
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

    template<typename T> struct is_no_storage_type_yet : boost::mpl::false_{};

    template <typename RegularStorageType>
    struct is_no_storage_type_yet<no_storage_type_yet<RegularStorageType> > : boost::mpl::true_ {};

    /**
       @brief stream operator, for debugging purpose
    */
    template <typename RST>
    std::ostream& operator<<(std::ostream& s, no_storage_type_yet<RST>) {
        return s << "no_storage_type_yet<" << RST() << ">" ;
    }

    /**
       \anchor descr_storage
       @brief main class for the basic storage

       We define here an important naming convention. We call:

       - the storages (or storage snapshots): are contiguous chunks of memory, accessed by 3 (by default, but not necessarily) indexes.
       These structures are univocally defined by 3 (by default) integers. These are currently 2 strides and the total size of the chunks. Note that (in 3D) the relation between these quantities
       (\f$stride_1\f$, \f$stride_2\f$ and \f$size\f$) and the dimensions x, y and z can be (depending on the storage layout chosen)
       \f[
       size=x*y*z \;;\;
       stride_2=x*y \;;\;
       stride_1=x .
       \f]
       The quantities \f$size\f$, \f$stride_2\f$ and \f$stride_1\f$ are arranged respectively in m_strides[0], m_strides[1], m_strides[2].
       - the \ref gridtools::storage_list "storage list": is a list of pointers (or snapshots) to storages. The snapshots are arranged on a 1D array. The \ref gridtools::accessor "accessor" class is
       responsible of computing the correct offests (relative to the given dimension) and address the storages correctly.
       - the \ref gridtools::data_field "data field": is a collection of storage lists, and can contain one or more storage lists of different sizes. It can be seen as a vector of vectors of storage pointers.
       (e.g. if the time T is the current dimension, 3 snapshots can be the fields at t, t+1, t+2)

       The base_storage class has a 1-1 relation with the storage concept, while the subclasses extend the concept of storage to the structure represented in the ASCII picture below.

       NOTE: the constraint of the snapshots accessed by the same data field are the following:
       - the memory layout (strides, space dimensions) is one for all the snapshots, and all the snapshots
       share the same iteration point
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
       This is achieved by using multiple inheritance.

       NOTE CUDA: It is important when subclassing from a storage object to reimplement the __device__ copy constructor, and possibly the method 'copy_data_to_gpu' which are used when cloning the class to the CUDA device.

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
                GRIDTOOLS_STATIC_ASSERT( boost::is_float<FloatType>::value, "The initialization value in the storage constructor must be a floating point number (e.g. 1.0). \nIf you want to store an integer you have to split construction and initialization \n(using the member \"initialize\"). This because otherwise the initialization value would be interpreted as an extra dimension");
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
                allocate();
            }

        template<typename ... UInt>
        void setup(UInt const& ... dims)
            {
                assign<space_dimensions-1>::apply(m_dims, std::tie(dims...));
                GRIDTOOLS_STATIC_ASSERT(sizeof...(UInt)==space_dimensions, "you tried to initialize a storage with a number of integer arguments different from its number of dimensions. This is not allowed. If you want to fake a lower dimensional storage, you have to add explicitly a \"1\" on the dimension you want to kill. Otherwise you can use a proper lower dimensional storage by defining the storage type using another layout_map.");

                GRIDTOOLS_STATIC_ASSERT(field_dimensions>0, "you specified a zero or negative value for a storage fields dimension");

                m_strides[0] = accumulate( multiplies(), dims...) ;
                _impl::assign_strides<(short_t)(space_dimensions-2), (short_t)(space_dimensions-1), layout>::apply(&m_strides[0], dims...);

#ifdef PEDANTIC
                //the following assert fails when we passed an argument to the arbitrary dimensional storage constructor which is not an unsigned integer (type uint_t).
                //You only have to pass the dimension sizes to this constructor, maybe you have to explicitly cast the value
                GRIDTOOLS_STATIC_ASSERT(accumulate(logical_and(), sizeof(UInt) == sizeof(uint_t) ... ), "You can disable this assertion by recompiling with the DISABLE_PEDANTIC flag set. This assert fails when we pass one or more arguments to the arbitrary dimensional storage constructor which are not of unsigned integer type (type uint_t). You can only pass the dimension sizes to this constructor." );

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
	      setup(dim1, dim2, dim3);
              m_fields[0]=pointer_type(ptr, size(), true);
              if(FieldDimension>1)
                  allocate(FieldDimension, 1);
	      set_name(s);
            }

        /**@brief destructor: frees the pointers to the data fields which are not managed outside */
        virtual ~base_storage(){
            for(ushort_t i=0; i<field_dimensions; ++i)
                // if(!m_fields[i].externally_managed())
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

        void allocate(ushort_t const& dims=FieldDimension, ushort_t const& offset=0){
            assert(dims>offset);
            is_set=true;
            for(ushort_t i=0; i<dims; ++i)
                m_fields[i+offset]=pointer_type(size());
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
        uint_t size() const { //cast to uint_t
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
        static constexpr int_t strides(StridesVector const& RESTRICT strides_){
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
            GRIDTOOLS_STATIC_ASSERT(iter::pos::value==sizeof...(UInt), "you have to pass in arguments of uint_t type");
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
        void increment( int_t* RESTRICT index_, StridesVector const& RESTRICT strides_/*, typename boost::enable_if_c< (layout::template pos_<Coordinate>::value >= 0) >::type* dummy=0*/){
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
        template <uint_t Coordinate, typename StridesVector>
        GT_FUNCTION
        void increment(int_t const& steps_, int_t* RESTRICT index_, StridesVector const& RESTRICT strides_){
#ifdef PEDANTIC
            GRIDTOOLS_STATIC_ASSERT(Coordinate < space_dimensions, "you have a storage in the iteration space whoose dimension is lower than the iteration space dimension. This might not be a problem, since trying to increment a nonexisting dimension has no effect. In case you want this feature comment out this assert.");

#endif
            if( layout::template at_< Coordinate >::value >= 0 )//static if
            {
#ifdef CXX11_ENABLED
                GRIDTOOLS_STATIC_ASSERT(StridesVector::size()==space_dimensions-1, "error: trying to compute the storage index using strides from another storage which does not have the same space dimensions. Are you explicitly incrementing the iteration space by calling base_storage::increment?");
#endif
                    *index_ += strides<Coordinate>(strides_)*steps_;
            }
        }

        GT_FUNCTION
        void set_index(uint_t value, int_t* index){
            *index=value;
        }

        template <uint_t Coordinate, typename StridesVector >
        GT_FUNCTION
        void initialize(uint_t const& steps_, uint_t const& /*block*/, int_t* RESTRICT index_, StridesVector const& RESTRICT strides_){
            //BOOST_STATIC_ASSERT(Coordinate < space_dimensions);

            if( Coordinate < space_dimensions && layout::template at_< Coordinate >::value >= 0 )//static if
            {
#ifdef CXX11_ENABLED
                GRIDTOOLS_STATIC_ASSERT(StridesVector::size()==space_dimensions-1, "error: trying to compute the storage index using strides from another storages which does not have the same space dimensions. Sre you explicitly initializing the iteration space by calling base_storage::initialize?");
#endif
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
        int_t const& strides(ushort_t i) const {
            // m_strides[0] contains the whole storage dimension."
            assert(i!=0);
            return m_strides[i];
        }

        /**@brief returns the storage strides
         */
        GT_FUNCTION
        int_t const* strides() const {
            GRIDTOOLS_STATIC_ASSERT(space_dimensions>1, "one dimensional storage");
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
        array<int_t, space_dimensions> m_strides;

    };

template < typename PointerType,
               typename Layout,
               bool IsTemporary,
               short_t FieldDimension>
const short_t base_storage<PointerType, Layout, IsTemporary, FieldDimension>::field_dimensions;

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

} //namespace gridtools
