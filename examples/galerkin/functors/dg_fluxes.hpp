#pragma once
#include "project_on_boundary.hpp"

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
            meta_storage_base<__COUNTER__,layout_map<2,1,0>,false> indexing{static_int<3>(), static_int<3>(), static_int<3>()};
#else
#ifdef NDEBUG
            constexpr
#endif
            meta_storage_base<__COUNTER__,layout_map<2,1,0>,false> indexing{Geometry::geo_map::order+1, Geometry::geo_map::order+1, Geometry::geo_map::order+1};

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
            meta_storage_base<__COUNTER__,layout_map<2,1,0>,false> indexing{static_int<3>(), static_int<3>(), static_int<3>()};
#else
#ifdef NDEBUG
            constexpr
#endif
                meta_storage_base<__COUNTER__,layout_map<2,1,0>,false> indexing{Geometry::geo_map::order+1, Geometry::geo_map::order+1, Geometry::geo_map::order+1};

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


    // [upwind]
    /**
       @class functor assembling the upwind flux

     */
    struct upwind {

        using in=accessor<0, enumtype::in, extent<> , 4>;
        using beta_n=accessor<1, enumtype::in, extent<> , 5>;
        using bd_mass_uu=accessor<2, enumtype::in, extent<> , 6>;
        using bd_mass_uv=accessor<3, enumtype::in, extent<> , 6>;
        using out=accessor<4, enumtype::inout, extent<> , 4> ;
        using arg_list=boost::mpl::vector<in, beta_n, bd_mass_uu, bd_mass_uv, out> ;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            dimension<1>::Index i;
            dimension<2>::Index j;
            dimension<3>::Index k;
            dimension<4>::Index row;
            dimension<5>::Index face;

            //definitions for the matrix accessor
            dimension<5>::Index col;
            dimension<6>::Index Mface;


            uint_t const num_faces=eval.get().template get_storage_dims<5>(bd_mass_uu());
            uint_t const n_dofs=eval.get().template get_storage_dims<4>(bd_mass_uu());

            // for all dofs in a boundary face (supposing that the dofs per face are the same)
            // important hypothesis: there are dofs on the faces, and they coincide with the ones for
            // the 3D element.
            // Otherwise everything gets way more complicated.

            //NOTE: we only loop on the 3 faces touching (0,0,0)
            for(short_t face1_ : {0,3,4})
            {
                /**see the face numbering:

                   index:                  .____.
                   0 (0,-1,0)             /  0 /|
                   1 (1,0,0)             .____. |5
                   2 (0,1,0)             |    |3.          z
                   3 (-1,0,0)           1|  4 |/       x__/
                   4 (0,0,-1)            .____.           |
                   5 (0,0,1)               2              y

                   NOTE: each element computes the fluxes only on faces 0,3,4

                */
                short_t face_opposite_ =
                    face1_==0?2
                    : face1_==1?3
                    : face1_==2?0
                    : face1_==3?1
                    : face1_==4?5
                    : face1_==5?4
                    : -666;


                /////////FOR DEBUGGING//////////////
                double bn=
                    face1_==0?0.
                    : face1_==1?1.
                    : face1_==2?0.
                    : face1_==3?-1.
                    : face1_==4?0.
                    : face1_==5?0.
                    : -666.;

                short_t opposite_i = (face1_==1)?1:(face1_==3)?-1:0;
                short_t opposite_j = (face1_==2)?1:(face1_==0)?-1:0;
                short_t opposite_k = (face1_==5)?1:(face1_==4)?-1:0;
                // hypothesis: if basis functions are localized n_dofs should be
                // the #dofs on a face
                for(short_t dof1_=0; dof1_<n_dofs; dof1_++)
                {//hypothesis: same #dofs on both faces
                    for(short_t dof2_=0; dof2_<n_dofs; dof2_++)
                    {
                        if (eval(beta_n(row+dof1_, face+face1_))>1e-15){ // outflow
                            auto tmp1 = eval(in(row+dof2_));
                            auto tmp2 = eval(bd_mass_uu(row+dof1_, col+dof2_, Mface+face1_));
                            auto tmp3 = eval(bd_mass_uv(row+dof1_, col+dof2_, Mface+face1_));
                            std::cout<<tmp1<<" * "<<tmp2<<" - "<<tmp1<<" * "<<tmp3<<"\n";
                            //take the contribution from the current elem: beta_n*Muu
                            eval(out(row+dof1_)) +=  eval(
                                // beta_n(row+dof1_, face+face1_) *
                                in(row+dof2_)
                                *bd_mass_uu(row+dof1_, col+dof2_, Mface+face1_)
                                -
                                beta_n(row+dof1_, face+face1_) *
                                in(row+dof2_)
                                *bd_mass_uv(row+dof1_, col+dof2_, Mface+face1_));
                        }
                        else if(eval(beta_n(row+dof1_, face+face1_))<-1e-15) // inflow
                        {
                            //take the contribution from the opposite face
                            eval(out(row+dof1_)) += eval(
                                // beta_n(row+dof1_, face+face_opposite_) *
                                in(opposite_i, opposite_j, opposite_k, dof2_)
                                *bd_mass_uu(opposite_i, opposite_j, opposite_k, dof1_, dof2_, face_opposite_)
                                -
                                beta_n(row+dof1_, face+face_opposite_) * in(opposite_i, opposite_j, opposite_k, dof2_)
                                *bd_mass_uv(opposite_i, opposite_j, opposite_k, dof1_, dof2_, face_opposite_));
                        }

                    }
                }
            }
        }
    };
            // [upwind]


}//namespace functors
