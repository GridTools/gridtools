#define DAWN_BACKEND_T CXXNAIVE
#ifndef BOOST_RESULT_OF_USE_TR1
#define BOOST_RESULT_OF_USE_TR1 1
#endif
#ifndef BOOST_NO_CXX11_DECLTYPE
#define BOOST_NO_CXX11_DECLTYPE 1
#endif
#ifndef GRIDTOOLS_DAWN_HALO_EXTENT
#define GRIDTOOLS_DAWN_HALO_EXTENT 3
#endif
#ifndef BOOST_PP_VARIADICS
#define BOOST_PP_VARIADICS 1
#endif
#ifndef BOOST_FUSION_DONT_USE_PREPROCESSED_FILES
#define BOOST_FUSION_DONT_USE_PREPROCESSED_FILES 1
#endif
#ifndef BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS 1
#endif
#ifndef GT_VECTOR_LIMIT_SIZE
#define GT_VECTOR_LIMIT_SIZE 30
#endif
#ifndef BOOST_FUSION_INVOKE_MAX_ARITY
#define BOOST_FUSION_INVOKE_MAX_ARITY GT_VECTOR_LIMIT_SIZE
#endif
#ifndef FUSION_MAX_VECTOR_SIZE
#define FUSION_MAX_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#ifndef FUSION_MAX_MAP_SIZE
#define FUSION_MAX_MAP_SIZE GT_VECTOR_LIMIT_SIZE
#endif
#ifndef BOOST_MPL_LIMIT_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE GT_VECTOR_LIMIT_SIZE
#endif

#include "codegen_api.hpp"

#include <gridtools/storage/cpu_ifirst.hpp>

auto const storage_builder = gridtools::storage::builder<gridtools::storage::cpu_ifirst>.type<float_type>();

class hori_diff_stencil {
    domain m_dom;
    decltype(storage_builder.id<3>().dimensions(0, 0, 0)()) m_lap;

    /*
        struct stencil_36 {
            // Members

            using tmp_halo_t = gridtools::halo<GRIDTOOLS_DAWN_HALO_EXTENT, GRIDTOOLS_DAWN_HALO_EXTENT, 0>;
            using tmp_meta_data_t = storage_traits_t::storage_info_t<0, 3, tmp_halo_t>;
            using tmp_storage_t = storage_traits_t::data_store_t<float_type, tmp_meta_data_t>;

            // Input/Output storages
            tmp_meta_data_t m_tmp_meta_data;
            tmp_storage_t m_lap;

          public:
            stencil_36(const domain &dom_)
                : m_dom(dom_), m_tmp_meta_data(dom_.size() + 1, dom_.size() + 1, dom_.ksize() + 2),
                  m_lap(m_tmp_meta_data) {}

            template <typename T_u, typename T_out>
            void run(T_u &u_, T_out &out_) {
                int iMin = m_dom.iminus();
                int iMax = m_dom.isize() - m_dom.iplus() - 1;
                int jMin = m_dom.jminus();
                int jMax = m_dom.jsize() - m_dom.jplus() - 1;
                int kMin = m_dom.kminus();
                int kMax = m_dom.ksize() - m_dom.kplus() - 1;
                {
                    gridtools::data_view<storage_ijk_t> u = gridtools::make_host_view(u_);
                    std::array<int, 3> u_offsets{0, 0, 0};
                    gridtools::data_view<storage_ijk_t> out = gridtools::make_host_view(out_);
                    std::array<int, 3> out_offsets{0, 0, 0};
                    gridtools::data_view<tmp_storage_t> lap = gridtools::make_host_view(m_lap);
                    std::array<int, 3> lap_offsets{0, 0, 0};
                    for (int k = kMin + 0 + 0; k <= kMax + 0 + 0; ++k) {
                        for (int i = iMin + -1; i <= iMax + 1; ++i) {
                            for (int j = jMin + -1; j <= jMax + 1; ++j) {
                                lap(i + 0, j + 0, k + 0) =
                                    ((((u(i + 1, j + 0, k + 0) + u(i + -1, j + 0, k + 0)) + u(i + 0, j + 1, k + 0)) +
                                         u(i + 0, j + -1, k + 0)) -
                                        ((::dawn::float_type)4 * u(i + 0, j + 0, k + 0)));
                            }
                        }
                        for (int i = iMin + 0; i <= iMax + 0; ++i) {
                            for (int j = jMin + 0; j <= jMax + 0; ++j) {
                                out(i + 0, j + 0, k + 0) =
                                    ((((lap(i + 1, j + 0, k + 0) + lap(i + -1, j + 0, k + 0)) + lap(i + 0, j + 1, k +
       0)) + lap(i + 0, j + -1, k + 0)) -
                                        ((::dawn::float_type)4 * lap(i + 0, j + 0, k + 0)));
                            }
                        }
                    }
                }
            }
        };
        */

  public:
    hori_diff_stencil(domain const &dom)
        : m_dom(dom), m_lap(storage_builder.id<3>().dimensions(gridtools::at_key<dim::i>(dom.size),
                          gridtools::at_key<dim::j>(dom.size),
                          gridtools::at_key<dim::k>(dom.size))()) {}

    template <class U, class Out>
    void operator()(U &&u, Out &&out) {
        //        m_stencil_36.run(u, out);
    }
};

hori_diff_stencil make_hori_diff_stencil(domain const &dom) { return {dom}; }
