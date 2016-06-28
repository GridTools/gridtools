/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include <boost/mpl/bool.hpp>

/**
@file
@brief dummy pointer object

this class is supposed to be replaced by (or to wrap) a smart pointer of our choice.
For the moment it just replaces a raw pointer

*/

namespace gridtools {

    /**
       @brief class wrapping a raw pointer
    */
    template < typename T >
    struct pointer {

      private:
        T *m_t;

      public:
        typedef T value_type;

        /**
           @brief default constructor
         */
        GT_FUNCTION
        pointer() : m_t(0) {}

        /**
           @brief construct from raw pointer
         */
        template < typename U >
        GT_FUNCTION pointer(U *t_)
            : m_t(t_) {
            assert(m_t);
        }

        /**
           @brief assign operator
         */
        template < typename U >
        GT_FUNCTION void operator=(U *t_) {
            m_t = t_;
        }

        /**
           @brief assign operator
         */
        GT_FUNCTION void operator=(pointer< T > const &other_) { m_t = other_.m_t; }

        /**
           @brief assign operator

           if cast between T and U is allowed
         */
        template < typename U >
        GT_FUNCTION void operator=(pointer< U > other_) {
            *m_t = *other_.get();
        }

        /**
           @brief returns the raw pointer (even if it's null)
        */
        GT_FUNCTION
        T *get() const { return m_t; }

        /**
           @brief access operator
         */
        GT_FUNCTION
        T *operator->() const {
            assert(m_t);
            return m_t;
        }

        /**
           @brief dereference operator
         */
        GT_FUNCTION
        T &operator*() const {
            assert(m_t);
            return *m_t;
        }

        /**
          @brief destroy pointer
         */
        GT_FUNCTION
        void destroy() {
            assert(m_t);
            delete m_t;
            m_t = NULL;
        }
    };

    template < typename T >
    pointer< T > make_pointer(T &t) {
        return pointer< T >(&t);
    }

    /**@brief deleting the pointers

       NOTE: this is called in the finalize stage of the gridtools computation,
       to delete the instances of the storage_info class
     */
    struct delete_pointer {

        delete_pointer() {}

        template < typename U >
        void operator()(U t) const {
            if (t.get())
                delete t.get();
        }
    };
}
