#pragma once

namespace functors{
    // [bassi rebay]
    /**
       @class functor assembling the Bassi-Rebay flux

     */
    template<typename Geometry>
    struct bassi_rebay {

        using geo_map=typename Geometry::geo_map;

        using in1=accessor<0, enumtype::in, extent<> , 5>;
        using in2=accessor<1, enumtype::in, extent<> , 5>;
        using out=accessor<2, enumtype::inout, extent<> , 5> ;
        using arg_list=boost::mpl::vector<in1, in2, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<1>::Index i;
            dimension<2>::Index j;
            dimension<3>::Index k;
            dimension<4>::Index row;


            //hypothesis here: the cardinaxlity is order^3 (isotropic 3D tensor product element)
#ifdef __CUDACC__
#ifdef NDEBUG
            constexpr
#endif
            meta_storage_base<__COUNTER__,layout_map<0,1,2>,false> indexing{static_int<3>(), static_int<3>(), static_int<3>()};
#else
#ifdef NDEBUG
            constexpr
#endif
            meta_storage_base<__COUNTER__,layout_map<0,1,2>,false> indexing{Geometry::geo_map::order+1, Geometry::geo_map::order+1, Geometry::geo_map::order+1};

#endif

            //for all dofs in a boundary face (supposing that the dofs per face are the same)
            for(short_t I=0; I<indexing.template dims<0>(); I++)
                for(short_t J=0; J<indexing.template dims<1>(); J++)
                {

                    //for each (3) faces
                    auto dof_x=indexing.index(0, (int)I, (int)J);
                    auto dof_xx=indexing.index(indexing.template dims<0>()-1, I, J);
                    auto dof_y=indexing.index(I, 0, J);
                    auto dof_yy=indexing.index(I, indexing.template dims<1>()-1, J);
                    auto dof_z=indexing.index(I, J, 0);
                    auto dof_zz=indexing.index(I, J, indexing.template dims<2>()-1);

                    //average the contribution from elem i-1 on the opposite face
                    eval(out(row+dof_x)) += (eval(in1(row+dof_x)) + eval(in2(i-1, row+dof_xx)))/2.;

                    //average the contribution from elem j-1 on the opposite face
                    eval(out(row+dof_y)) += (eval(in1(row+dof_y)) + eval(in2(j-1, row+dof_yy)))/2.;

                    //average the contribution from elem k-1 on the opposite face
                    eval(out(row+dof_z)) += (eval(in1(row+dof_z)) + eval(in2(k-1, row+dof_zz)))/2.;

                }
        }
    };
    // [bassi rebay]

    // [lax friedrich]
    /**
       @class functor assembling the Lax-Friedrich flux

     */
    template<typename Geometry, class Flux>
    struct lax_friedrich {

        using geo_map=typename Geometry::geo_map;

        using in=accessor<0, enumtype::in, extent<> , 4>;
        using out=accessor<1, enumtype::inout, extent<> , 4> ;
        using arg_list=boost::mpl::vector<in, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<1>::Index i;
            dimension<2>::Index j;
            dimension<3>::Index k;
            dimension<4>::Index row;


            //hypothesis here: the cardinaxlity is order^3 (isotropic 3D tensor product element)
#ifdef __CUDACC__
#ifdef NDEBUG
            constexpr
#endif
            meta_storage_base<__COUNTER__,layout_map<0,1,2>,false> indexing{static_int<3>(), static_int<3>(), static_int<3>()};
#else
#ifdef NDEBUG
            constexpr
#endif
                meta_storage_base<__COUNTER__,layout_map<0,1,2>,false> indexing{Geometry::geo_map::order+1, Geometry::geo_map::order+1, Geometry::geo_map::order+1};

#endif

            //for all dofs in a boundary face (supposing that the dofs per face are the same)
            for(short_t I=0; I<indexing.template dims<0>(); I++)
                for(short_t J=0; J<indexing.template dims<1>(); J++)
                {

                    //for each (3) faces
                    auto dof_x=indexing.index(0, (int)I, (int)J);
                    auto dof_xx=indexing.index(indexing.template dims<0>()-1, I, J);
                    auto dof_y=indexing.index(I, 0, J);
                    auto dof_yy=indexing.index(I, indexing.template dims<1>()-1, J);
                    auto dof_z=indexing.index(I, J, 0);
                    auto dof_zz=indexing.index(I, J, indexing.template dims<2>()-1);

                    const auto N=eval.get().template get_storage_dims<3>(in());

                    //initial value
                    auto c=eval(D(Flux()(in())));

                    //find the maximum
                    for(ushort_t it_=1; it_<N; ++it_)
                        c=(c>eval(expressions::D(Flux()(in(row+it_))))) ? c : eval(D(Flux()(in(row+it_))));

                    //average the contribution from elem i-1 on the opposite face
                    eval(out(row+dof_x)) += (eval(Flux()(in(row+dof_x))+Flux()(in(i-1, row+dof_xx)))/2. - c*(eval(in(row+dof_x)) - eval(in(i-1, row+dof_xx))))/2.;

                    //average the contribution from elem j-1 on the opposite face
                    eval(out(row+dof_x)) += (eval(Flux()(in(row+dof_y))+Flux()(in(i-1, row+dof_yy))) - c*eval(in(row+dof_y)) - eval(in(i-1, row+dof_yy)))/2.;

                    //average the contribution from elem k-1 on the opposite face
                    eval(out(row+dof_x)) += (eval(Flux()(in(row+dof_z))+Flux()(in(i-1, row+dof_zz))) - c*(eval(in(row+dof_z)) - eval(in(i-1, row+dof_zz))))/2.;

                }
        }
    };
    // [lax friedrich]


}//namespace functors
