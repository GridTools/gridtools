#pragma once
#include <boost/mpl/bool.hpp>

/**
@file
@brief dummy pointer object

this class is supposed to be replaced by (or to wrap) a smart pointer of our choice.
For the moment it just replaces a raw pointer

*/

namespace gridtools{

    /**
       @brief class wrapping a raw pointer
    */
    template<typename T>
    struct pointer{

    private:
        T const* m_t;
    public:
        typedef T value_type;

        /**
           @brief default constructor
         */
        GT_FUNCTION
        pointer(): m_t(0){}

        /**
           @brief construct from raw pointer
         */
        template <typename U>
        GT_FUNCTION
        pointer(U const* t_): m_t(t_){}

        /**
           @brief assign operator
         */
        template <typename U>
        GT_FUNCTION
        void operator = (U const* t_){
            m_t=t_;
        }

        /**
           @brief returns the raw pointer (even if it's null)
        */
        GT_FUNCTION
        T const* get() const {return m_t;}

        /**
           @brief access operator
         */
        GT_FUNCTION
        T const* operator -> () const {
            assert(m_t);
            return m_t;
        }

        /**
           @brief dereference operator
         */
        GT_FUNCTION
        T const& operator * () const {
            assert (m_t);
            return *m_t;
        }

    };



    /**@brief deleting the pointers

       NOTE: this is called in the finalize stage of the gridtools computation,
       to delete the instances of the storage_info class
     */
    struct delete_pointer{

        delete_pointer() {}

        template<typename U>
        void operator()(U t) const{
            delete t.get();
        }
    };


    /** \addtogroup specializations Specializations
        Partial specializations
        @{
    */
    template<typename T>
    struct is_ptr_to_tmp : boost::mpl::false_{};

    template<typename T>
    struct is_ptr_to_tmp<pointer<const T> > : boost::mpl::bool_<T::is_temporary> {};
    /**@}*/
}
