#pragma once
namespace gridtools{

    template < typename T >
    struct is_any_iterate_domain_storage;

    template <typename BaseStorage>
    struct intrepid_storage;

    template < typename BaseStorage >
    struct is_any_iterate_domain_storage<intrepid_storage<BaseStorage> > : boost::mpl::true_{};


    template<int T>
    struct operator_selector;

    template<>
    struct operator_selector<1>{
        template <typename T>
        static typename T::value_type& apply(T* t, int i){ return (*t)(i);}
    };

    template<>
    struct operator_selector<2>{
        template <typename T>
        static typename T::value_type& apply(T* t, int i){ return (*t)(i,0);}

        template <typename T>
        static  typename T::value_type& apply(T* t, int i, int j){ return (*t)(i,j);}
    };


    template<>
    struct operator_selector<3>{
        template <typename T>
        static typename T::value_type& apply(T* t, int i){ return (*t)(i,0,0);}

        template <typename T>
        static typename T::value_type& apply(T* t, int i, int j){ return (*t)(i,j,0);}

        template <typename T>
        static typename T::value_type& apply(T* t, int i, int j, int k){ return (*t)(i,j,k);}
};

    /**decorator for the regular storage to be used instead of a Intrepid::FieldContainer*/
    template <typename Storage>
    class intrepid_storage // : public Storage
    {
    private :
        Storage & m_storage;

    public:
        // using Storage::Storage;
        using storage_t=Storage;
        // GRIDTOOLS_STATIC_ASSERT(
        //     boost::mpl::fold<
        //     Storage::layout::layout_vector,
        //     boost::mpl::bool_<0>,
        //     boost::mpl::greater<boost::mpl::_2, boost::mpl::_1> >::type::value,
        //     "the memory layout for this storage must be increasing strides (layout_map<0,1,2,3>)"
        //     )

        intrepid_storage(storage_t& storage_) : m_storage(storage_){}

        Storage /*const*/& get_storage(){ return m_storage;}


        template<typename ... UInt>
        typename storage_t::value_type const& operator()(UInt ... indices) const {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(UInt)<=Storage::space_dimensions, "accessing the storage using too many indices")
                return operator_selector<Storage::space_dimensions>::apply(&m_storage, indices ...);
            // GRIDTOOLS_STATIC_ASSERT(sizeof...(UInt)>=Storage::space_dimensions, "accessing the storage using too few indices")
        }

        template <typename ... UInt>
        typename Storage::value_type& operator()(UInt ... i) {
            return operator_selector<Storage::space_dimensions>::apply(&m_storage, i ...);
        }

        int size() const {
            return (int) m_storage.size();
        }

        int dimension(int k) const {
            return (int) m_storage.dims(k);
        }

        // //bad things happening
        // typename Storage::value_type const& operator[](int k) const {
        //     return m_storage.fields()[0].get()[k];
        // }

        // //very bad things happening
        // typename Storage::value_type& operator[](int k){
        //     return m_storage.fields()[0].get()[k];
        // }

    };
}//namespace gridtools
