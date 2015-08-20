#pragma once
#include "base_storage.hpp"

namespace gridtools{
    /** @brief storage class containing a buffer of data snapshots

        it is a list of \ref gridtools::base_storage "storages"

        \include storage.dox

    */
    template < typename Storage, short_t ExtraWidth>
    struct storage_list : public Storage
    {

        typedef storage_list<Storage, ExtraWidth> type;
        /*If the following assertion fails, you probably set one field dimension to contain zero (or negative) snapshots. Each field dimension must contain one or more snapshots.*/
        GRIDTOOLS_STATIC_ASSERT(ExtraWidth>0, "you probably set one field dimension to contain zero (or negative) snapshots. Each field dimension must contain one or more snapshots.");
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
}//namespace gridtools
