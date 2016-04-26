#pragma once

namespace gridtools{

    template <typename Storage, uint_t Size>
    struct expandable_parameters : field<Storage, Size>::type {

        typedef typename field<Storage, Size>::type super;
        typedef Storage basic_type;
        using super::data_field;
        // using type_tt = super::type_tt;

    public:

        // template<typename MetaData, typename = typename boost::enable_if<typename is_storage_info<MetaData>::type, int >::type >
        // expandable_parameters(MetaData const& meta_) : super(meta_) {
        // }

        // public methods:
        /**
           @brief assign a chunk of the pointers array from a large storage list to a smaller one (i.e. this one).

           @param a the larger storage field than this one
           @offset the offsets at which the copy starts
         */
        template<ushort_t OtherSize>
        void assign_pointers(expandable_parameters<Storage, OtherSize>& other, uint_t offset){
            GRIDTOOLS_STATIC_ASSERT((OtherSize >= Size), "Cannot assign pointers from a smaller storage");
            for(ushort_t i; i<Size; ++i)
                if(i<OtherSize)
                    this->m_fields[i]=other.fields()[offset+i];
        }

        /**
           @brief assignment operator: assigns the first pointers of the input storage list
           to the fields data member

           Used with boost::fusion::copy, when copying the boost::fusion::vector of storage pointers
           the operator= gets called on all storage. Since we are copying from a large storage list to
           a small one, and we want to start from the beginning of the long storage list, this operator
           suits our need.
        */
        template<ushort_t OtherSize>
        void operator=(expandable_parameters<Storage, OtherSize> const& other){

            GRIDTOOLS_STATIC_ASSERT((OtherSize >= Size), "Cannot assign pointers from a smaller storage");
            for(ushort_t i; i<Size; ++i)
                this->m_fields[i]=other.fields()[i];
        }

        template<typename ... UInt>
        typename super::value_type& operator()(uint_t const& dim, UInt const & ... idx) {
            assert(this->m_meta_data.index(idx ...) < this->m_meta_data.size());
            assert(this->is_set);
            return (this->m_fields[dim])[this->m_meta_data.index(idx ...)];
        }

        /**
           @brief copy constructor doing casts between expandable parameters of different sizes

           A larger storage list can be casted to a smaller one, not vice versa.
         */
        template<ushort_t OtherSize>
        expandable_parameters(expandable_parameters<Storage, OtherSize> const& other){

            GRIDTOOLS_STATIC_ASSERT((OtherSize >= Size), "Cannot assign pointers from a smaller storage");
            for(ushort_t i; i<Size; ++i)
                this->m_fields[i]=other.fields()[i];
        }

    };

    template <typename Storage, uint_t Size>
    struct is_storage<expandable_parameters<Storage, Size> >: boost::mpl::true_{};
}//namespace gridtools
