#pragma once
#include "../gridtools.hpp"

/**
@file
@brief dummy pointer object

this class is supposed to be replaced by (or to wrap) a smart pointer of our choice.
For the moment it just replace a raw pointer

NOTE: one reason for it being needed is that
fusion set does not support raw pointers as keys. Instead of creating a set of pairs
I went for the pointer wrapper
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
        GT_FUNCTION
        pointer(): m_t(0){}

        template <typename U>
        GT_FUNCTION
        pointer(U const* t_): m_t(t_){}

        template <typename U>
        GT_FUNCTION
        void operator = (U const* t_){
            m_t=t_;
        }

        /**
           @brief returns the raw pointer
        */
        GT_FUNCTION
        T const* get() {return m_t;}
        GT_FUNCTION
        T const* operator -> () const {return m_t;}
        GT_FUNCTION
        T const& operator * () const {return *m_t;}

    };



    /**@brief deleting the pointers

       NOTE: this is called in the finalize stage of the gridtools computation
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
