/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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
        GT_FUNCTION pointer(T *t_) : m_t(t_) { assert(m_t); }

        /**
           @brief copy constructor
         */
        GT_FUNCTION pointer(pointer< T > const &other_) : m_t(other_.m_t) {}

        GT_FUNCTION ~pointer() { m_t = NULL; }

        /**
           @brief assign operator
         */
        GT_FUNCTION void operator=(T *t_) { m_t = t_; }

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
