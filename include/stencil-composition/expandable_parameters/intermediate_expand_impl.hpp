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

namespace gridtools {
    namespace _impl {
        template < typename T, typename Vec, ushort_t ID, bool val >
        struct new_storage;

        /** new storage when the storage is not temporary*/
        template < typename T, typename Vec, ushort_t ID >
        struct new_storage< T, Vec, ID, true > {

            template < typename DomFull >
            static typename boost::remove_reference<
                typename boost::fusion::result_of::at< Vec, static_ushort< ID > >::type >::type::value_type *
            apply(DomFull const &dom_full_) {
                return new typename boost::remove_reference<
                    typename boost::fusion::result_of::at< Vec, static_ushort< ID > >::type >::type::
                    value_type(dom_full_.template storage_pointer< arg< ID, std::vector< pointer< T > > > >()
                                   ->at(0)
                                   ->meta_data(),
                        "expandable params",
                        false /*do_allocate*/);
            }
        };

        /** when the storage is temporary return nullptr instead of a new storage*/
        template < typename T, typename Vec, ushort_t ID >
        struct new_storage< T, Vec, ID, false > {

            template < typename DomFull >
            static typename boost::remove_reference<
                typename boost::fusion::result_of::at< Vec, static_ushort< ID > >::type >::type::value_type *
            apply(DomFull const &) {
                return nullptr;
            }
        };

        /**
           @brief functor used to initialize the storage in a boost::fusion::vector full an
           instance of gridtools::domain_type
        */
        template < typename DomainFull, typename Vec >
        struct initialize_storage {

          private:
            DomainFull const &m_dom_full;
            Vec &m_vec_to;

          public:
            initialize_storage(DomainFull const &dom_full_, Vec &vec_to_) : m_dom_full(dom_full_), m_vec_to(vec_to_) {}

            /**
               @brief initialize the storage vector, specialization for the expandable args
             */
            template < ushort_t ID, typename T >
            void operator()(arg< ID, std::vector< pointer< T > > >) {

                boost::fusion::at< static_ushort< ID > >(m_vec_to) = new_storage< T,
                    Vec,
                    ID,
                    !boost::remove_reference< decltype(
                        m_dom_full.template storage_pointer< arg< ID, std::vector< pointer< T > > > >()->at(
                            0)) >::type::value_type::is_temporary >::apply(m_dom_full);
            }

            /**
               @brief initialize the storage vector, specialization for the normal args
             */
            template < ushort_t ID, typename Storage >
            void operator()(arg< ID, Storage >) {
                // copy the gridtools pointer
                boost::fusion::at< static_ushort< ID > >(m_vec_to) =
                    m_dom_full.template storage_pointer< arg< ID, Storage > >();
            }
        };

        template < typename Domain >
        struct check_length {

          private:
            Domain &m_domain;
            uint_t m_size;

          public:
            check_length(Domain &dom_, uint_t size_) : m_domain(dom_), m_size(size_) {}

            template < typename Arg >
            void operator()(Arg) const {
                // error here means that the sizes of the expandable parameter lists do not match
                if (!is_temporary_storage< typename boost::mpl::at< typename Domain::arg_list_mpl,
                        typename Arg::index_type >::type >::value)
                    assert(
                        boost::fusion::at< typename Arg::index_type >(m_domain.m_storage_pointers)->size() == m_size);
            }
        };

        /**
           @brief functor used to delete the storages containing the chunk of pointers
        */
        template < typename Vec >
        struct delete_storage {

          private:
            Vec &m_vec_to;

          public:
            delete_storage(Vec &vec_to_) : m_vec_to(vec_to_) {}

            template < typename T >
            void operator()(T) {
                // unset the storage, so that it does not try to release the pointers it contains
                boost::fusion::at< typename T::index_type >(m_vec_to)->storage_pointer()->unset();
                // filtering out temporary storages
                if (!boost::fusion::at< typename T::index_type >(m_vec_to)->is_temporary) {
                    delete_pointer deleter;
                    deleter(boost::fusion::at< typename T::index_type >(m_vec_to));
                }
            }
        };

        /**
           @brief functor used to assign the next chunk of storage pointers
        */
        template < typename DomainFull, typename DomainChunk >
        struct prepare_expandable_params {

          private:
            DomainFull const &m_dom_full;
            DomainChunk &m_dom_chunk;
            uint_t const &m_idx;

          public:
            prepare_expandable_params(DomainFull const &dom_full_, DomainChunk &dom_chunk_, uint_t const &i_)
                : m_dom_full(dom_full_), m_dom_chunk(dom_chunk_), m_idx(i_) {}

            template < ushort_t ID, typename T >
            void operator()(arg< ID, std::vector< pointer< T > > >) {

                if (!is_temporary_storage<
                        typename boost::mpl::at_c< typename DomainChunk::arg_list_mpl, ID >::type >::value) {
                    // the vector of pointers
                    pointer< std::vector< pointer< T > > > const &ptr_full_ =
                        m_dom_full.template storage_pointer< arg< ID, std::vector< pointer< T > > > >();
                    auto ptr_chunk_ = boost::fusion::at< static_ushort< ID > >(m_dom_chunk.m_storage_pointers);
                    // reset the pointer to the host version, since they'll be accessed from the host
                    (*(ptr_chunk_->storage_pointer())).set(*ptr_full_, m_idx);
                    ptr_chunk_->set_on_host();
                    // update the device pointers (TODO: should not copy the heavy data)
                    ptr_chunk_->clone_to_device();
                }
            }
        };

        /**
           @brief functor used to assign the next chunk of storage pointers
        */
        template < typename Backend, typename DomainFull, typename DomainChunk >
        struct assign_expandable_params {

          private:
            DomainFull const &m_dom_full;
            DomainChunk &m_dom_chunk;
            uint_t const &m_idx;

          public:
            assign_expandable_params(DomainFull const &dom_full_, DomainChunk &dom_chunk_, uint_t const &i_)
                : m_dom_full(dom_full_), m_dom_chunk(dom_chunk_), m_idx(i_) {}

            template < ushort_t ID, typename T >
            void operator()(arg< ID, std::vector< pointer< T > > >) {

                if (!is_temporary_storage<
                        typename boost::mpl::at_c< typename DomainChunk::arg_list_mpl, ID >::type >::value) {
                    // the vector of pointers
                    pointer< std::vector< pointer< T > > > const &ptr_full_ =
                        m_dom_full.template storage_pointer< arg< ID, std::vector< pointer< T > > > >();

                    auto ptr_chunk_ = boost::fusion::at< static_ushort< ID > >(m_dom_chunk.m_storage_pointers);

#ifndef NDEBUG
                    if (!ptr_chunk_.get() || !ptr_full_.get()) {
                        printf("The storage pointer is already null. Did you call finalize too early?");
                        assert(false);
                    }
#endif
                    (*(ptr_chunk_->storage_pointer())).set(*ptr_full_, m_idx);
                    if (Backend::s_backend_id == enumtype::Cuda) {
                        ptr_chunk_->set_on_host();
                        ptr_chunk_->h2d_update();
                    }
                }
            }
        };

        /**
           @brief functor used to assign the next chunk of storage pointers
        */
        template < typename Backend, typename DomainFull >
        struct finalize_expandable_params {

          private:
            DomainFull const &m_dom_full;

          public:
            finalize_expandable_params(DomainFull const &dom_full_) : m_dom_full(dom_full_) {}

            template < ushort_t ID, typename T >
            void operator()(arg< ID, std::vector< pointer< T > > >) {

                auto ptr = boost::fusion::at< static_ushort< ID > >(m_dom_full.m_storage_pointers);
                if (ptr.get()) { // if it's a temporary it might have been freed already
                    for (auto &&i : *ptr) {
                        // hard-setting the on_device flag for the hybrid_pointers:
                        // since the storages used get created on-the-fly the original storages do
                        // not know that they are still on the device
                        if (Backend::s_backend_id == enumtype::Cuda) {
                            i->set_on_device();
                            i->storage_pointer()->set_on_device();
                            i->d2h_update();
                        }
                    }
                }
            }
        };
    } // namespace _impl
} // namespace gridtools
