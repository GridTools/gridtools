#pragma once
namespace gdl{

template <typename Assembler, typename Fe, typename BCFunctor, typename FluxFunctor>
struct bc_apply{

private:
    Assembler & m_assembler;
    Fe & m_fe;

public:
    bc_apply(Assembler & assembler_, Fe & fe_ ) :
        m_assembler(assembler_),
        m_fe(fe_)
    {}

    template <typename Grid, typename Storage1, typename Storage2>
    #ifdef __CUDACC__
    gt::stencil*
    #else
    std::shared_ptr< gt::stencil >
    #endif
    compute(Grid const& grid_, Storage1 & bc_, Storage2 & tr_bc_){

        struct transform {
            typedef  gt::arg<0, typename Assembler::storage_type >    p_jac_det;
            typedef  gt::arg<1, typename Assembler::geometry_t::weights_storage_t >   p_weights;
            typedef  gt::arg<2, typename Fe::basis_function_storage_t> p_phi;
            typedef  gt::arg<3, Storage1 > p_bc;
            typedef  gt::arg<4, Storage2 > p_bc_integrated;
        };

        //adding an extra index at the end of the layout_map
        // using bc_storage_info_t=storage_info< __COUNTER__, gt::layout_map_union<Layout, gt::layout_map<0> > >;
        // using bc_storage_t = storage_t< bc_storage_info_t >;

        typedef typename boost::mpl::vector< typename transform::p_jac_det, typename transform::p_weights, typename transform::p_phi, typename transform::p_bc, typename transform::p_bc_integrated> mpl_list_transform_bc;

        gt::aggregator_type<mpl_list_transform_bc> domain_transform_bc(
            boost::fusion::make_vector(
                &m_assembler.jac_det()
                ,&m_assembler.fe_backend().cub_weights()
                ,&m_fe.val()
                ,&bc_
                ,&tr_bc_
                ));

        auto transform_bc=gt::make_computation< BACKEND >(
            domain_transform_bc, grid_
            , gt::make_multistage(
            enumtype::execute<enumtype::forward>()
            , gt::make_stage< BCFunctor >( typename transform::p_bc(), typename transform::p_bc() )
            , gt::make_stage< functors::transform >( typename transform::p_jac_det(), typename transform::p_weights(), typename transform::p_phi(), typename transform::p_bc(), typename transform::p_bc_integrated() )
                )
            );
        return transform_bc;
    }




    template <typename Grid, typename Storage1, typename Storage2, typename Storage3, typename Storage4, typename Storage5, typename Storage6>

    #ifdef __CUDACC__
    gt::stencil*
    #else
    std::shared_ptr< gt::stencil >
    #endif
    apply(Grid const& grid_, Storage1& tr_bc_, Storage2& result_, Storage3 & bd_beta_n_, Storage4& bd_mass_, Storage5& bd_mass_uv_, Storage6 & u_){


        struct bc{
            typedef  gt::arg<0, Storage1 > p_bc;
            typedef  gt::arg<1, Storage2 > p_result;
            typedef  gt::arg<2, Storage3 > p_beta_n;
            typedef  gt::arg<3, Storage4 > p_bd_mass_uu;
            typedef  gt::arg<4, Storage5 > p_bd_mass_uv;
        };

        typedef typename boost::mpl::vector< typename bc::p_bc, typename bc::p_result, typename bc::p_beta_n, typename bc::p_bd_mass_uu, typename bc::p_bd_mass_uv> mpl_list_bc;

        gt::aggregator_type<mpl_list_bc> domain_apply_bc(boost::fusion::make_vector(
                                                         &tr_bc_
                                                         ,&result_
                                                         ,&bd_beta_n_
                                                         ,&bd_mass_
                                                         ,&bd_mass_uv_
                                                         ));

        auto apply_bc=gt::make_computation< BACKEND >(
            domain_apply_bc, grid_
            , gt::make_multistage(
                enumtype::execute<enumtype::forward>()
                , gt::make_stage< FluxFunctor >(typename bc::p_bc(), typename bc::p_beta_n(), typename bc::p_bd_mass_uu(), typename bc::p_bd_mass_uv(),  typename bc::p_result())
                )
            );

        return apply_bc;
    }
};
}//namespace gdl
