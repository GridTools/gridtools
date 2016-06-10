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
#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include "stencil_composition/stencil_composition.hpp"
#include <boost/current_function.hpp>

using namespace gridtools;
using namespace enumtype;

uint_t count;
bool result;

struct print_ {
    print_(void)
    {}

    template <typename T>
    void operator()(T const& v) const {
        if (T::value != count)
            result = false;
        ++count;
    }
};

struct print_plchld {
    mutable uint_t count;
    mutable bool result;

    print_plchld(void)
    {}

    template <typename T>
    void operator()(T const& v) const {
        if (T::index_type::value != count) {
            result = false;
        }
        ++count;
    }
};

bool test_domain_indices() {

    typedef backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::storage_type< float_type,
        backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::storage_info< 0, layout_map< 0, 1, 2 > > >::type
        storage_type;
    typedef backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::temporary_storage_type< float_type,
        backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::storage_info< 0, layout_map< 0, 1, 2 > > >::type
        tmp_storage_type;

    uint_t d1 = 10;
    uint_t d2 = 10;
    uint_t d3 = 10;

    backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type in(meta_,-1., "in");
    storage_type out(meta_,-7.3, "out");
    storage_type coeff(meta_,8., "coeff");

    typedef arg<2, tmp_storage_type > p_lap;
    typedef arg<1, tmp_storage_type > p_flx;
    typedef arg<5, tmp_storage_type > p_fly;
    typedef arg<0, storage_type > p_coeff;
    typedef arg<3, storage_type > p_in;
    typedef arg<4, storage_type > p_out;

    result = true;

    typedef boost::mpl::vector<p_lap, p_flx, p_fly, p_coeff, p_in, p_out> accessor_list;

    aggregator_type<accessor_list> domain
       (boost::fusion::make_vector(&out, &in, &coeff /*,&fly, &flx*/));

    count = 0;
    result = true;

    print_plchld pfph;
    count = 0;
    result = true;
    boost::mpl::for_each<aggregator_type<accessor_list>::placeholders>(pfph);


    return result;
}
