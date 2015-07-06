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


    template<>
    struct operator_selector<4>{
        template <typename T>
        static typename T::value_type& apply(T* t, int i){ return (*t)(i,0,0,0);}

        template <typename T>
        static typename T::value_type& apply(T* t, int i, int j){ return (*t)(i,j,0,0);}

        template <typename T>
        static typename T::value_type& apply(T* t, int i, int j, int k){ return (*t)(i,j,k,0);}

        template <typename T>
        static typename T::value_type& apply(T* t, int i, int j, int k, int l){ return (*t)(i,j,k,l);}
    };

    /**decorator for the regular storage to be used instead of a Intrepid::FieldContainer*/
    template <typename Storage>
    class intrepid_storage
    {
    private :
        Storage& m_storage;
        uint_t m_rank;

    public:
        using storage_t=Storage;

        intrepid_storage(storage_t& storage_, uint_t rank_=0) : m_storage(storage_), m_rank(rank_){}

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

        int rank() const {
            return (int) m_rank? m_rank : Storage::space_dimensions;
        }

    };
}//namespace gridtools
