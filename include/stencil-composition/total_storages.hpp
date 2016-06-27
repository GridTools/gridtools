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

namespace gridtools {

    /**@brief metafunction that counts the total number of data fields which are neceassary for this functor (i.e.
     * number of storage
     * instances times number of fields per storage)
    */
    template < typename StoragesVector, int_t EndIndex >
    struct total_storages {
        DISALLOW_COPY_AND_ASSIGN(total_storages);
        // the index must not exceed the number of storages
        GRIDTOOLS_STATIC_ASSERT(EndIndex <= boost::mpl::size< StoragesVector >::type::value,
            "the index must not exceed the number of storages");

        template < typename Index_ >
        struct get_field_dimensions {
            typedef typename boost::mpl::int_<
                boost::mpl::at< StoragesVector, Index_ >::type::value_type::field_dimensions >::type type;
        };

        typedef typename boost::mpl::if_c<
            (EndIndex < 0),
            boost::mpl::int_< 0 >,
            typename boost::mpl::fold< typename reversed_range< uint_t, 0, EndIndex >::type,
                boost::mpl::int_< 0 >,
                boost::mpl::plus< boost::mpl::_1, get_field_dimensions< boost::mpl::_2 > > >::type >::type type;

        static const uint_t value = type::value;

      private:
        total_storages();
    };
} // namespace gridtools
