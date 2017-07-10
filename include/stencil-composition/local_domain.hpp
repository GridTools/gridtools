/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/zip.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/utility.hpp>
#include <iosfwd>

#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../common/gpu_clone.hpp"
#include "../common/host_device.hpp"
#include "../common/is_temporary_storage.hpp"
#include "../common/generic_metafunctions/fusion_vector_check_bound.hpp"
#include "arg.hpp"
#include "esf.hpp"
#include "storage_wrapper.hpp"
#include "tile.hpp"

#include <boost/fusion/include/as_set.hpp>

namespace gridtools {

    namespace {
        template < class T, size_t N >
        GT_FUNCTION constexpr size_t get_size(T(&)[N]) {
            return N;
        }

        template < typename T, typename V, unsigned N = (boost::mpl::size< T >::value - 1) >
        GT_FUNCTION typename boost::enable_if_c< (N == 0), void >::type copy_ptrs(T &t, V &other) {
            auto &left = boost::fusion::at_c< N >(t).second;
            auto &right = boost::fusion::at_c< N >(other).second;
            for (unsigned i = 0; i < get_size(left); ++i) {
                left[i] = right[i];
            }
        }

        template < typename T, typename V, unsigned N = (boost::mpl::size< T >::value - 1) >
        GT_FUNCTION typename boost::enable_if_c< (N > 0), void >::type copy_ptrs(T &t, V &other) {
            auto &left = boost::fusion::at_c< N >(t).second;
            auto &right = boost::fusion::at_c< N >(other).second;
            for (unsigned i = 0; i < get_size(left); ++i) {
                left[i] = right[i];
            }
            copy_ptrs< T, V, N - 1 >(t, other);
        }
    }

    /**
     * This is the base class for local_domains to extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. There is one version which provide grid to the functor
     * and one that does not
     *
     */
    template < typename T >
    struct local_domain_base;

    template < typename SWL, typename E, bool I >
    class local_domain;

    template < typename StorageWrapperList, typename EsfArgs, bool IsStateful >
    struct local_domain_base< local_domain< StorageWrapperList, EsfArgs, IsStateful > >
        : public clonable_to_gpu< local_domain< StorageWrapperList, EsfArgs, IsStateful > > {

        typedef local_domain< StorageWrapperList, EsfArgs, IsStateful > derived_t;

        typedef local_domain_base< derived_t > this_type;

        typedef EsfArgs esf_args;

        typedef StorageWrapperList storage_wrapper_list_t;

        typedef typename max_i_extent_from_storage_wrapper_list< storage_wrapper_list_t >::type max_i_extent_t;

        typedef typename boost::mpl::fold< StorageWrapperList,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1,
                                               boost::fusion::pair< arg_from_storage_wrapper< boost::mpl::_2 >,
                                                   data_ptr_from_storage_wrapper< boost::mpl::_2 > > > >::type
            arg_to_data_ptr_map_t;

        typedef typename boost::mpl::fold<
            StorageWrapperList,
            boost::mpl::vector0<>,
            boost::mpl::if_<
                boost::mpl::contains< boost::mpl::_1,
                    boost::add_pointer< boost::add_const< storage_info_from_storage_wrapper< boost::mpl::_2 > > > >,
                boost::mpl::_1,
                boost::mpl::push_back< boost::mpl::_1,
                    boost::add_pointer< boost::add_const<
                        storage_info_from_storage_wrapper< boost::mpl::_2 > > > > > >::type storage_info_ptr_list;

        typedef typename boost::mpl::fold< StorageWrapperList,
            boost::mpl::map0<>,
            boost::mpl::insert< boost::mpl::_1,
                                               boost::mpl::pair< storage_info_from_storage_wrapper< boost::mpl::_2 >,
                                                   temporary_info_from_storage_wrapper< boost::mpl::_2 > > > >::type
            storage_info_tmp_info_t;

        typedef typename boost::fusion::result_of::as_map<
            typename boost::fusion::result_of::as_vector< arg_to_data_ptr_map_t >::type >::type data_ptr_fusion_map;
        typedef
            typename boost::fusion::result_of::as_vector< storage_info_ptr_list >::type storage_info_ptr_fusion_list;

        // get a storage from the list of storages
        template < typename IndexType >
        struct get_storage {
            typedef typename boost::mpl::at< StorageWrapperList, IndexType >::type storage_wrapper_t;
            typedef typename storage_wrapper_t::storage_t type;
            static_assert(
                !boost::is_same< boost::mpl::false_, type >::value, "Cannot find storage type in local_domain.");
        };

        // get a storage wrapper from the list of storages
        template < typename IndexType >
        struct get_storage_wrapper {
            typedef typename boost::mpl::at< StorageWrapperList, IndexType >::type storage_wrapper_t;
            typedef storage_wrapper_t type;
            static_assert(!boost::is_same< boost::mpl::false_, type >::value,
                "Cannot find storage wrapper type in local_domain.");
        };

        // get a storage from the list of storages
        template < typename IndexType >
        struct get_arg {
            typedef typename boost::mpl::at< StorageWrapperList, IndexType >::type storage_wrapper_t;
            typedef typename storage_wrapper_t::arg_t type;
            static_assert(!boost::is_same< boost::mpl::false_, type >::value, "Cannot find arg type in local_domain.");
        };

        //********** members *****************
        data_ptr_fusion_map m_local_data_ptrs;
        storage_info_ptr_fusion_list m_local_storage_info_ptrs;
        //********** end members *****************

        GT_FUNCTION_WARNING
        local_domain_base() {}

        GT_FUNCTION_DEVICE local_domain_base(local_domain_base const &other)
            : m_local_storage_info_ptrs(other.m_local_storage_info_ptrs) {
            copy_ptrs(m_local_data_ptrs, other.m_local_data_ptrs);
        }

        struct print_local_storage {
            std::ostream &out_s;
            print_local_storage(std::ostream &out_s) : out_s(out_s) {}

            template < typename T >
            void operator()(T const &e) const {
                typedef typename storage_wrapper_elem< typename boost::fusion::result_of::first< T >::type,
                    storage_wrapper_list_t >::type storage_wrapper_t;
                out_s << "arg_index: " << arg_index_from_storage_wrapper< storage_wrapper_t >::value << std::endl;
                for (unsigned i = 0; i < storage_wrapper_t::num_of_storages; ++i)
                    out_s << e.second[i] << "\t";
                out_s << "\n\n";
            }
        };

        struct print_local_storage_info {
            std::ostream &out_s;
            print_local_storage_info(std::ostream &out_s) : out_s(out_s) {}

            template < typename T >
            void operator()(T const &e) const {
                std::cout << "storage_info: " << e << std::endl;
            }
        };

        GT_FUNCTION
        void info(std::ostream &out_s) const {
            out_s << "        -----v SHOWING LOCAL ARGS BELOW HERE v-----\n";
            boost::fusion::for_each(m_local_data_ptrs, print_local_storage(out_s));
            boost::fusion::for_each(m_local_storage_info_ptrs, print_local_storage_info(out_s));
            out_s << "        -----^ SHOWING LOCAL ARGS ABOVE HERE ^-----\n";
        }
    };

    /**
     * This class extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. This version does not provide grid
     * to the function operator
     *
     * @tparam StoragePointers The mpl vector of the storage pointer types
     * @tparam MetaData The mpl vector of the meta data pointer types sequence
     * @tparam EsfArgs The mpl vector of the args (i.e. placeholders for the storages)
                       for the current ESF
     * @tparam IsStateful The flag stating if the local_domain is aware of the position in the iteration domain
     */
    template < typename StorageWrapperList, typename EsfArgs, bool IsStateful >
    struct local_domain : public local_domain_base< local_domain< StorageWrapperList, EsfArgs, IsStateful > > {

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< StorageWrapperList, is_storage_wrapper >::value),
            "Local domain contains wrong type for parameter StorageWrapperList");
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< EsfArgs, is_arg >::value), "Local domain contains wrong type for parameter EsfArgs");

        typedef local_domain_base< local_domain< StorageWrapperList, EsfArgs, IsStateful > > base_type;

        GT_FUNCTION
        local_domain() {}

        GT_FUNCTION_DEVICE local_domain(local_domain const &other) : base_type(other) {}

        /**stub methods*/
        GT_FUNCTION
        uint_t i() const { return 1e9; }
        GT_FUNCTION
        uint_t j() const { return 1e9; }
        GT_FUNCTION
        uint_t k() const { return 1e9; }
    };

    template < typename StorageWrapperList, typename EsfArgs, bool IsStateful >
    std::ostream &operator<<(std::ostream &s, local_domain< StorageWrapperList, EsfArgs, IsStateful > const &) {
        return s << "local_domain<stuff>";
    }

    template < typename T >
    struct is_local_domain : boost::mpl::false_ {};

    template < typename StorageWrapperList, typename EsfArgs, bool IsStateful >
    struct is_local_domain< local_domain< StorageWrapperList, EsfArgs, IsStateful > > : boost::mpl::true_ {};

    template < typename T >
    struct local_domain_is_stateful;

    template < typename StorageWrapperList, typename EsfArgs, bool IsStateful >
    struct local_domain_is_stateful< local_domain< StorageWrapperList, EsfArgs, IsStateful > >
        : boost::mpl::bool_< IsStateful > {};

    template < typename T >
    struct local_domain_esf_args;

    template < typename StorageWrapperList, typename EsfArgs, bool IsStateful >
    struct local_domain_esf_args< local_domain< StorageWrapperList, EsfArgs, IsStateful > > {
        typedef EsfArgs type;
    };

} // namespace gridtools
