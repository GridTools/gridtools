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
#ifndef _FIELD_ON_THE_FLY_H_
#define _FIELD_ON_THE_FLY_H_

namespace gridtools {
    /**
       Struct that contains the information for an array with halo. It
       construct all necessary information to execute halo_exchange_generic

       \tparam DataType Type of the elements stored in the array
       \tparam DIMS Number of dimensions of the array
       \tparam layoutmap Specification of the layout map of the data (as in halo_exchange_dynamic)
     */
    template < typename DataType, typename _t_layoutmap, template < typename > class Traits >
    struct field_on_the_fly : public Traits< DataType >::base_field {
        typedef typename reverse_map< _t_layoutmap >::type inner_layoutmap; // This is necessary since the internals of
                                                                            // gcl use "increasing stride order" instead
                                                                            // of "decreasing stride order"
        typedef _t_layoutmap outer_layoutmap;
        static const int DIMS = Traits< DataType >::I;

        typedef typename Traits< DataType >::base_field base_type;

        typedef field_on_the_fly< DataType, _t_layoutmap, Traits > this_type;

        typedef DataType value_type;

        mutable DataType *ptr;

        /** Default constructor
         */
        field_on_the_fly(){};

        template < typename T1 >
        field_on_the_fly< T1, _t_layoutmap, Traits > &retarget() {
            void *tmp = this;
            return *(reinterpret_cast< field_on_the_fly< T1, _t_layoutmap, Traits > * >(tmp));
        }

        template < typename T1 >
        field_on_the_fly< T1, _t_layoutmap, Traits > copy() const {
            const void *tmp = this;
            return *(reinterpret_cast< const field_on_the_fly< T1, _t_layoutmap, Traits > * >(tmp));
        }

        void set_pointer(DataType *pointer) { ptr = pointer; }

        DataType *get_pointer() const { return ptr; }

        /**
           Constructor that takes an gridtools::array of halo descriptors. The order
           of the elements are the logical order in which the user sees the
           dimensions. Layout map is used to permute the entries in the proper
           way.

           \param p Pointer to the array containing the data
           \param halos Array (gridtools::array) of array halos
         */
        field_on_the_fly(DataType *p, array< halo_descriptor, DIMS > const &halos) : ptr(p) {
            //        std::cout << "FOF                                          " << t_layoutmap() << " " <<
            //        layoutmap() << std::endl;

            for (int i = 0; i < DIMS; ++i) {
                base_type::add_halo(inner_layoutmap()[i],
                    halos[i].minus(),
                    halos[i].plus(),
                    halos[i].begin(),
                    halos[i].end(),
                    halos[i].total_length());
            }

            base_type::setup();
        }
        /**
           Method to explicitly create a field_on_the_fly. It takes an gridtools::array
           of halo descriptors. The order of the elements are the logical order in
           which the user sees the dimensions. Layout map is used to permute the
           entries in the proper way.

           \param p Pointer to the array containing the data
           \param halos Array (gridtools::array) of array halos
         */
        void create(DataType *p, array< halo_descriptor, DIMS > const &halos) {
            ptr = p;
            for (int i = 0; i < DIMS; ++i) {
                base_type::add_halo(inner_layoutmap()[i],
                    halos[i].minus(),
                    halos[i].plus(),
                    halos[i].begin(),
                    halos[i].end(),
                    halos[i].total_length());
            }

            base_type::setup();
        }

        // halo_descriptor const& operator[](int i) const {
        //   return (*this).halos[i];
        // }

        // template <typename new_value_type>
        // operator field_on_the_fly<new_value_type, layoutmap, Traits>*() {

        const DataType *the_pointer() const { return ptr; }
    };

    template < typename DataType, typename layoutmap, template < typename > class Traits >
    std::ostream &operator<<(std::ostream &s, field_on_the_fly< DataType, layoutmap, Traits > const &fot) {
        return s << static_cast< typename field_on_the_fly< DataType, layoutmap, Traits >::base_type >(fot) << " -> "
                 << reinterpret_cast< void const * >(fot.the_pointer());
    }

    // /**
    //    Struct that contains the information for an array with halo. It
    //    construct all necessary information to execute halo_exchange_generic

    //    \tparam DataType Type of the elements stored in the array
    //    \tparam DIMS Number of dimensions of the array
    //    \tparam layoutmap Specification of the layout map of the data (as in halo_exchange_dynamic)
    //  */
    // template <typename DataType,
    //           int DIMS,
    //           typename layoutmap=typename default_layout_map<DIMS>::type>
    // struct field_on_the_fly_man: empty_field<DataType, DIMS> {
    //   typedef empty_field_no_dt<DIMS> base_type;

    //   typedef DataType value_type;
    //   typedef layoutmap layout_map;
    //   DataType *ptr;

    //   /** Default constructor
    //    */
    //   field_on_the_fly_man() {};

    //   /**
    //      Constructor that takes an gridtools::array of halo descriptors. The order
    //      of the elements are the logical order in which the user sees the
    //      dimensions. Layout map is used to permute the entries in the proper
    //      way.

    //      \param p Pointer to the array containing the data
    //      \param halos Array (gridtools::array) of array halos
    //    */
    //   field_on_the_fly_man(DataType* p, array<halo_descriptor, DIMS> const & halos)
    //     : ptr(p)
    //   {
    //     for (int i=0; i<DIMS; ++i) {
    //       base_type::add_halo(layoutmap()[i],
    //                           halos[i].minus(),
    //                           halos[i].plus(),
    //                           halos[i].begin(),
    //                           halos[i].end(),
    //                           halos[i].total_length());
    //     }

    //     base_type::setup();
    //   }
    //   /**
    //      Method to explicitly create a field_on_the_fly. It takes an gridtools::array
    //      of halo descriptors. The order of the elements are the logical order in
    //      which the user sees the dimensions. Layout map is used to permute the
    //      entries in the proper way.

    //      \param p Pointer to the array containing the data
    //      \param halos Array (gridtools::array) of array halos
    //    */
    //   void create(DataType* p, array<halo_descriptor, DIMS> const & halos)
    //   {
    //     ptr = p;
    //     for (int i=0; i<DIMS; ++i) {
    //       base_type::add_halo(layoutmap()[i],
    //                           halos[i].minus(),
    //                           halos[i].plus(),
    //                           halos[i].begin(),
    //                           halos[i].end(),
    //                           halos[i].total_length());
    //     }

    //     base_type::setup();
    //   }

    // };
}

#endif
