#pragma once
#include <boost/fusion/include/as_set.hpp>
#include "common/generic_metafunctions/is_sequence_of.hpp"

/**
@file
@brief implementing a set
*/
namespace gridtools{

    /**
       @brief class that given a generic MPL sequence creates a fusion set.

       The interface of this class allows to insert and get elements of the sequence give its type.
       It also allows to query if the element corresponding to a given type has been or not initialized

       It is used to hold the list of meta-storages, in which a 1-1 relation is needed between instances
       and types.
     */
    template<typename Sequence>
    struct metadata_set{
        GRIDTOOLS_STATIC_ASSERT( boost::mpl::is_sequence<Sequence>::value,
                                 "internal error: not a sequence" );
        GRIDTOOLS_STATIC_ASSERT( (is_sequence_of<Sequence, is_pointer>::value),
                                 "internal error: not a sequence of pointers");
        typedef typename boost::fusion::result_of::as_set<Sequence>::type set_t;

    private:
        set_t m_set;

    public:

        GT_FUNCTION
        metadata_set() : m_set(){};

        /**
           @brief device copy constructor
         */
        __device__
        metadata_set(metadata_set const& other) : m_set(other.m_set){};

        /**
           @brief inserts a new instance in the sequence
           NOTE: pass by value
        */
        template <typename T>
        GT_FUNCTION
        void insert(T new_instance)
            {
                GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key<set_t, T>::type::value),
                                        "the type used for the lookup in the metadata set is not present in the set. Did you use the correct type as meta storage?");
                assert(!present<T>());//must be uninitialized
                boost::fusion::at_key<T>(m_set)=new_instance;
            }

        /**
           @brief returns the raw pointer given a key

           \tparam T key
        */
        template <typename T>
        GT_FUNCTION
        T const& get()
            {
                GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key<set_t, T>::type::value), "Internal error: calling metadata_set::get with a metadata type which has not been defined.");
                assert(present<T>());//must be initialized
                return boost::fusion::at_key<T>(m_set);
            }

        /**@brief returns the sequence by non-const reference*/
        GT_FUNCTION
        set_t& sequence_view() {return m_set;}

        /**@brief returns the sequence by const reference*/
        GT_FUNCTION
        set_t const& sequence_view() const {return m_set;}

        /**@bief queries if the given key corresponds to a pointer which has been initialized*/
        template <typename T>
        GT_FUNCTION
        bool present() {
            GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::has_key<set_t, T>::type::value), "internal error: calling metadata_set::present with a metadata type which has not been defined");
            return boost::fusion::at_key<T>(m_set).get();
        }


    };

    template <typename T>
    struct is_metadata_set : boost::mpl::false_{};

    template <typename T>
    struct is_metadata_set<metadata_set<T> > : boost::mpl::true_{};


    template <typename U>
    struct is_storage;

    /** inserts an element in the set if it is not present

        used for the metadata_set in the domain_type
    */
    template <typename Sequence, typename Arg>
    struct insert_if_not_present{

#ifdef PEDANTIC //disabling in case of generic accessors
        GRIDTOOLS_STATIC_ASSERT(is_storage<Arg>::type::value, "if you are using generic accessors disable the pedantic mode. Otherwise most probably you used in the domain_type constructor a storage type which is not supported.");
#endif
        GRIDTOOLS_STATIC_ASSERT(is_metadata_set<Sequence>::type::value, "wrong type");

    private :
        Sequence& m_seq;
        Arg const& m_arg;
    public:
        insert_if_not_present(Sequence& seq_, Arg const& arg_): m_seq(seq_), m_arg(arg_){}
        void operator()()const{
            if (!m_seq.template present< pointer<const typename Arg::meta_data_t> >())
                m_seq.insert(pointer<const typename Arg::meta_data_t>(&(m_arg.meta_data())));                 }
    };
}
