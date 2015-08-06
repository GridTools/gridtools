#pragma once

#include "base_storage.hpp"
#include "location_type.hpp"


namespace gridtools {

    /**
       The backend is, as usual, declaring what the storage types are
     */
    struct _backend {
    public:
        template <typename LocationType, typename X, typename LayoutMap>
        struct _storage_type;

        template <ushort_t NColors, char const* Name, typename X, typename LayoutMap>
        struct _storage_type<location_type<0, NColors, Name>, X, LayoutMap> {
            using type = base_storage<wrap_pointer<double>, LayoutMap, location_type<0, NColors, Name> >;
        };

        template <ushort_t NColors, char const* Name, typename X, typename LayoutMap>
        struct _storage_type<location_type<1, NColors, Name>, X, LayoutMap> {
            using type = base_storage<wrap_pointer<double>, LayoutMap, location_type<1, NColors, Name> >;
        };

        template <typename LocationType>
        using storage_type = typename _storage_type<LocationType, void, layout_map<0,1,2,3> >::type;

    };

    struct colored_backend : public _backend {
    private:

        template <ushort_t NColors, char const* Name, typename Accessor, typename Computation, typename Coords>
        static
        void
        dispatch_on_locationtype(location_type<0, NColors, Name>,
                                 Accessor & acc,
                                 Computation const x,
                                 Coords const & coords)
        {
            const auto low_bounds = acc.grid().ll_indices({coords.lb0, coords.lb1}, location_type<0, NColors, Name>());
            const auto high_bounds = acc.grid().ll_indices({coords.ub0, coords.ub1}, location_type<0, NColors, Name>());

            std::cout << "User Level Iteration (closed) \n"
                      << "    from " << coords.lb0 << " to " << coords.ub0 << " \n"
                      << "    from " << coords.lb1 << " to " << coords.ub1 << " "
                      << std::endl;

            std::cout << "Low bounds  " << low_bounds << std::endl;
            std::cout << "High bounds (closed) " << high_bounds << std::endl;
            std::cout << "Iteration space on Cells (closed) "
                      << "from " << low_bounds[0] << " to " <<  high_bounds[0] << " "
                      << "from " << low_bounds[1] << " to " <<  high_bounds[1] << " "
                      << "from " << low_bounds[2] << " to " <<  high_bounds[2] << " "
                      << std::endl;
            for (int i = low_bounds[0]; i <= high_bounds[0]; ++i) {
                for (int j = low_bounds[1]; j <= high_bounds[1]; ++j) { // they should always be 0 and 1 for cells
                    acc.template set_ll_ijk<location_type<0, NColors, Name> >(i, j, low_bounds[2]);
                    for (int k = low_bounds[2]; k <= high_bounds[2]; ++k) {
                        typename decltype(x)::functor()(acc);
                        acc.inc_ll_k();
                    }
                }
            }
        }

        template <ushort_t NColors, char* Name, typename Accessor, typename Computation, typename Coords>
        static
        void
        dispatch_on_locationtype(location_type<1, NColors, Name>,
                                 Accessor & acc,
                                 Computation const x,
                                 Coords const & coords)
        {
            const auto low_bounds = acc.grid().ll_indices({coords.lb0, coords.lb1}, location_type<1, NColors, Name>());
            const auto high_bounds = acc.grid().ll_indices({coords.ub0-1, coords.ub1-1}, location_type<1, NColors, Name>());
            std::cout << "Low bounds  " << low_bounds << std::endl;
            std::cout << "High bounds " << high_bounds << std::endl;
            std::cout << "Iteration space on Edges "
                      << "from " << low_bounds[0] << " to " <<  high_bounds[0] << " "
                      << "from " << low_bounds[1] << " to " <<  high_bounds[1] << " "
                      << "from " << low_bounds[2] << " to " <<  high_bounds[2] << " "
                      << std::endl;
            for (int i = low_bounds[0]; i <= high_bounds[0]; ++i) {
                for (int j = low_bounds[1]; j <= high_bounds[1]; ++j) { // they should always be 0 and 1 for cells
                    acc.template set_ll_ijk<location_type<1, NColors, Name> >(i, j, low_bounds[2]);
                    for (int k = low_bounds[2]; k <= high_bounds[2]; ++k) {
                        typename decltype(x)::functor()(acc);
                        acc.inc_ll_k();
                    }
                }
            }
        }

    public:

        /**
           run needs to dispatch to the right implementation depending on
           location types. In this particular example the two implementations
           are identical, but this is not the general case.
         */
        template <typename Accessor, typename Computation, typename Coords>
        static
        void
        run(Accessor & acc, Computation const x, Coords const & coords)
        {
            dispatch_on_locationtype(typename Accessor::location_type(),
                                     acc, x, coords);
        }

    };

} // namespace gridtools
