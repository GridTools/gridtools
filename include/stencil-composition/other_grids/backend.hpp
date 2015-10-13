#pragma once

#include "storage/storage.hpp"
#include "storage/meta_storage.hpp"
#include "location_type.hpp"
#include "stencil-composition/backend_base.hpp"
#include "storage/wrap_pointer.hpp"

namespace gridtools {

    /**
       The backend is, as usual, declaring what the storage types are
     */
    template<enumtype::backend BackendId, enumtype::strategy StrategyType >
    struct backend : public backend_base<BackendId, StrategyType>{
    public:
        // template <typename LocationType, typename X, typename LayoutMap>
        // struct _storage_type;

        // template <ushort_t NColors, typename X, typename LayoutMap>
        // struct _storage_type<location_type<0, NColors>, X, LayoutMap> {
        //     using type = base_storage<wrap_pointer<double>, LayoutMap, location_type<0, NColors> >;
        // };

        // template <ushort_t NColors, typename X, typename LayoutMap>
        // struct _storage_type<location_type<1, NColors>, X, LayoutMap> {
        //     using type = base_storage<wrap_pointer<double>, LayoutMap, location_type<1, NColors> >;
        // };

        // template <ushort_t NColors, typename X, typename LayoutMap>
        // struct _storage_type<location_type<2, NColors>, X, LayoutMap> {
        //     using type = base_storage<wrap_pointer<double>, LayoutMap, location_type<2, NColors> >;
        // };

        typedef backend_base<BackendId, StrategyType> base_t;

        using base_t::backend_traits_t;
        using base_t::strategy_traits_t;

        static const enumtype::strategy s_strategy_id=base_t::s_strategy_id;
        static const enumtype::backend s_backend_id =base_t::s_backend_id;

        template <typename LocationType>
        using meta_storage_t = storage_info<LocationType::value, layout_map<0,1,2,3> >;

        template <typename LocationType, typename ValueType>
        using storage_t = storage< base_storage<wrap_pointer<ValueType>, meta_storage_t<LocationType>, 1> >;

    };

    struct colored_backend : public backend<enumtype::Host, enumtype::Block> {
    private:

        template <ushort_t NColors, typename Accessor, typename Computation, typename Coords>
        static
        void
        dispatch_on_locationtype(location_type<0, NColors>,
                                 Accessor & acc,
                                 Computation const x,
                                 Coords const & coords)
        {
            const auto low_bounds = acc.grid().ll_indices({coords.lb0, coords.lb1, coords.lb2},
                                                          location_type<0, NColors>());
            const auto high_bounds = acc.grid().ll_indices({coords.ub0-1, coords.ub1-1, coords.ub2-1},
                                                           location_type<0, NColors>());

            std::cout << "User Level Iteration (closed) \n"
                      << "    from " << coords.lb0 << " to " << coords.ub0-1 << " (closed)\n"
                      << "    from " << coords.lb1 << " to " << coords.ub1-1 << " (closed)\n"
                      << "    from " << coords.lb2 << " to " << coords.ub2-1 << " (closed)"
                      << std::endl;

            std::cout << "Low bounds  " << low_bounds << std::endl;
            std::cout << "High bounds (closed) " << high_bounds << std::endl;
            std::cout << "Iteration space on Cells (closed) "
                      << "from " << low_bounds[0] << " to " <<  high_bounds[0] << " "
                      << "from " << low_bounds[1] << " to " <<  high_bounds[1] << " "
                      << "from " << low_bounds[2] << " to " <<  high_bounds[2] << " "
                      << "from " << low_bounds[3] << " to " <<  high_bounds[3] << " "
                      << std::endl;
            for (int i = low_bounds[0]; i <= high_bounds[0]; ++i) {
                for (int j = low_bounds[1]; j <= high_bounds[1]; ++j) { // they should always be 0 and 1 for cells
                    for (int k = low_bounds[2]; k <= high_bounds[2]; ++k) {
                        acc.template set_ll_ijk<location_type<0, NColors> >(i, j, k, low_bounds[3]);
                        for (int l = low_bounds[3]; l <= high_bounds[3]; ++l) {
                            typename decltype(x)::functor()(acc);
                            acc.template inc_ll<3>();
                        }
                    }
                }
            }
        }

        template <ushort_t NColors, typename Accessor, typename Computation, typename Coords>
        static
        void
        dispatch_on_locationtype(location_type<1, NColors>,
                                 Accessor & acc,
                                 Computation const x,
                                 Coords const & coords)
        {
            const auto low_bounds = acc.grid().ll_indices({coords.lb0, coords.lb1, coords.lb2},
                                                          location_type<1, NColors>());
            const auto high_bounds = acc.grid().ll_indices({coords.ub0-1, coords.ub1-1, coords.ub2-1},
                                                           location_type<1, NColors>());

            std::cout << "User Level Iteration (closed) \n"
                      << "    from " << coords.lb0 << " to " << coords.ub0-1 << " (closed)\n"
                      << "    from " << coords.lb1 << " to " << coords.ub1-1 << " (closed)\n"
                      << "    from " << coords.lb2 << " to " << coords.ub2-1 << " (closed)"
                      << std::endl;

            std::cout << "Low bounds  " << low_bounds << std::endl;
            std::cout << "High bounds (closed) " << high_bounds << std::endl;
            std::cout << "Iteration space on Cells (closed) "
                      << "from " << low_bounds[0] << " to " <<  high_bounds[0] << " "
                      << "from " << low_bounds[1] << " to " <<  high_bounds[1] << " "
                      << "from " << low_bounds[2] << " to " <<  high_bounds[2] << " "
                      << "from " << low_bounds[3] << " to " <<  high_bounds[3] << " "
                      << std::endl;
            for (int i = low_bounds[0]; i <= high_bounds[0]; ++i) {
                for (int j = low_bounds[1]; j <= high_bounds[1]; ++j) { // they should always be 0 and 1 for cells
                    for (int k = low_bounds[2]; k <= high_bounds[2]; ++k) {
                        acc.template set_ll_ijk<location_type<1, NColors> >(i, j, k, low_bounds[3]);
                        for (int l = low_bounds[3]; l <= high_bounds[3]; ++l) {
                            typename decltype(x)::functor()(acc);
                            acc.template inc_ll<3>();
                        }
                    }
                }
            }
        }

        template <ushort_t NColors, typename Accessor, typename Computation, typename Coords>
        static
        void
        dispatch_on_locationtype(location_type<2, NColors>,
                                 Accessor & acc,
                                 Computation const x,
                                 Coords const & coords)
        {
            static_assert(true, "WRONG");
            // const auto low_bounds = acc.grid().ll_indices({coords.lb0, coords.lb1}, location_type<2, NColors>());
            // const auto high_bounds = acc.grid().ll_indices({coords.ub0-1, coords.ub1-1}, location_type<2, NColors>());
            // std::cout << "Low bounds  " << low_bounds << std::endl;
            // std::cout << "High bounds " << high_bounds << std::endl;
            // std::cout << "Iteration space on vertexes "
            //           << "from " << low_bounds[0] << " to " <<  high_bounds[0] << " "
            //           << "from " << low_bounds[1] << " to " <<  high_bounds[1] << " "
            //           << "from " << low_bounds[2] << " to " <<  high_bounds[2] << " "
            //           << std::endl;
            // for (int i = low_bounds[0]; i <= high_bounds[0]; ++i) {
            //     for (int j = low_bounds[1]; j <= high_bounds[1]; ++j) { // they should always be 0 and 1 for cells
            //         acc.template set_ll_ijk<location_type<2, NColors> >(i, j, low_bounds[2]);
            //         for (int k = low_bounds[2]; k <= high_bounds[2]; ++k) {
            //             typename decltype(x)::functor()(acc);
            //             acc.inc_ll_k();
            //         }
            //     }
            //
        }


    public:

        /**
           run needs to dispatch to the right implementation depending on
           location types. In this particular example the two implementations
           are identical, but this is not the general case.
         */
        template <typename IterateDomain, typename Computation, typename Coords>
        static
        void
        run(IterateDomain & it, Computation const x, Coords const & coords)
        {
            typedef typename IterateDomain::location_type location_type_t;
            dispatch_on_locationtype(location_type_t(),
                                     it, x, coords);
        }

    };

} // namespace gridtools
