/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

/*
 * This example tests the internal stencil serialization!
 * For normal usage of Serialbox see: https://github.com/MeteoSwiss-APN/serialbox
 */

#if defined(USE_SERIALBOX)

#include "serialization_setup.hpp"

using namespace gridtools;
using namespace enumtype;

typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

struct copy_to_temporaries {
    typedef accessor< 0, enumtype::in, extent<>, 3 > in;
    typedef accessor< 1, enumtype::inout, extent<>, 3 > tmp1;
    typedef accessor< 2, enumtype::inout, extent<>, 3 > tmp2;

    typedef boost::mpl::vector< in, tmp1, tmp2 > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        eval(tmp1()) = 0.25 * eval(in());
        eval(tmp2()) = 0.75 * eval(in());
    }
};

struct copy_from_temporaries {
    typedef accessor< 0, enumtype::in, extent<>, 3 > tmp1;
    typedef accessor< 1, enumtype::in, extent<>, 3 > tmp2;
    typedef accessor< 2, enumtype::inout, extent<>, 3 > out;
    typedef boost::mpl::vector< tmp1, tmp2, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        eval(out()) = eval(tmp1()) + eval(tmp2());
    }
};

TEST_F(serialization_setup, temporaries) {
    namespace ser = serialbox::gridtools;
    uint_t d1 = 5, d2 = 6, d3 = 7;

    // Storages
    storage_t &in = make_storage("in", d1, d2, d3);
    storage_t &out = make_storage("out", d1, d2, d3);

    for_each("in", [&in](int i, int j, int k) { in(i, j, k) = i + j + k; });
    for_each("out", [&out](int i, int j, int k) { out(i, j, k) = -1; });

    // Domain
    typedef arg< 0, temporary_storage_t > p_tmp1;
    typedef arg< 1, temporary_storage_t > p_tmp2;
    typedef arg< 2, storage_t > p_in;
    typedef arg< 3, storage_t > p_out;
    typedef boost::mpl::vector< p_tmp1, p_tmp2, p_in, p_out > accessor_list;

    gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out));

    // Grid
    uint_t halo_size = 1;
    uint_t di[5] = {halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
    uint_t dj[5] = {halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    auto copy_with_temporaries =
        make_computation< backend_t >(domain,
            grid,
            make_multistage(execute< forward >(),
                                          make_stage< copy_to_temporaries >(p_in(), p_tmp1(), p_tmp2()),
                                          make_stage< copy_from_temporaries >(p_tmp1(), p_tmp2(), p_out())));

    try {
        // Run & serialize
        // ===============
        ser::serializer serializer(ser::open_mode::Write, directory(), prefix());
        copy_with_temporaries->ready();
        copy_with_temporaries->steady();
        copy_with_temporaries->run(serializer, "copy_with_temporaries");

        // Verify
        // ======
        ser::serializer ref_serializer(ser::open_mode::Read, directory(), prefix());

        // Check functor
        ASSERT_TRUE(verify_storages(in, out, grid, halo_size));

        // Check field meta-information
        ASSERT_TRUE(verify_field_meta_info(ref_serializer));
        ASSERT_TRUE(ref_serializer.has_field("tmp_0"));
        ASSERT_TRUE(ref_serializer.has_field("tmp_1"));

        // Check serialized data
        ASSERT_EQ(ref_serializer.savepoints().size(), 4);
        ASSERT_EQ(ref_serializer.fieldnames().size(), 4);
        ASSERT_TRUE(ref_serializer.global_meta_info().has_key("stencils"));
        ASSERT_EQ(ref_serializer.get_global_meta_info_as< std::vector< std::string > >("stencils").size(), 1);
        ASSERT_EQ(ref_serializer.get_global_meta_info_as< std::vector< std::string > >("stencils")[0],
            "copy_with_temporaries");

        // Load serialized data
        // ====================

        // Savepoint: input of "copy_to_temporaries"
        storage_t &copy_to_temporaries_input_in = make_storage("copy_to_temporaries_input_in", d1, d2, d3);
        ref_serializer.read("in", ref_serializer.savepoints()[0], copy_to_temporaries_input_in);

        // Savepoint: output of "copy_to_temporaries"
        storage_t &copy_to_temporaries_output_in = make_storage("copy_to_temporaries_output_in", d1, d2, d3);
        storage_t &copy_to_temporaries_output_tmp_0 = make_storage("copy_to_temporaries_output_tmp_0", d1, d2, d3);
        storage_t &copy_to_temporaries_output_tmp_1 = make_storage("copy_to_temporaries_output_tmp_1", d1, d2, d3);
        ref_serializer.read("in", ref_serializer.savepoints()[1], copy_to_temporaries_output_in);
        ref_serializer.read("tmp_0", ref_serializer.savepoints()[1], copy_to_temporaries_output_tmp_0);
        ref_serializer.read("tmp_1", ref_serializer.savepoints()[1], copy_to_temporaries_output_tmp_1);

        // Savepoint: input of "copy_from_temporaries"
        storage_t &copy_from_temporaries_input_out = make_storage("copy_from_temporaries_input_out", d1, d2, d3);
        storage_t &copy_from_temporaries_input_tmp_0 = make_storage("copy_from_temporaries_input_tmp_0", d1, d2, d3);
        storage_t &copy_from_temporaries_input_tmp_1 = make_storage("copy_from_temporaries_input_tmp_1", d1, d2, d3);
        ref_serializer.read("out", ref_serializer.savepoints()[2], copy_from_temporaries_input_out);
        ref_serializer.read("tmp_0", ref_serializer.savepoints()[2], copy_from_temporaries_input_tmp_0);
        ref_serializer.read("tmp_1", ref_serializer.savepoints()[2], copy_from_temporaries_input_tmp_1);

        // Savepoint: output of "copy_from_temporaries"
        storage_t &copy_from_temporaries_output_out = make_storage("copy_from_temporaries_output_out", d1, d2, d3);
        storage_t &copy_from_temporaries_output_tmp_0 = make_storage("copy_from_temporaries_output_tmp_0", d1, d2, d3);
        storage_t &copy_from_temporaries_output_tmp_1 = make_storage("copy_from_temporaries_output_tmp_1", d1, d2, d3);
        ref_serializer.read("out", ref_serializer.savepoints()[3], copy_from_temporaries_output_out);
        ref_serializer.read("tmp_0", ref_serializer.savepoints()[3], copy_from_temporaries_output_tmp_0);
        ref_serializer.read("tmp_1", ref_serializer.savepoints()[3], copy_from_temporaries_output_tmp_1);

        // Verify serialized data
        // ======================

        // Verify: "copy_to_temporaries"
        storage_t &copy_to_temporaries_output_tmp_0_ref =
            make_storage("copy_to_temporaries_output_tmp_0_ref", d1, d2, d3);
        storage_t &copy_to_temporaries_output_tmp_1_ref =
            make_storage("copy_to_temporaries_output_tmp_1_ref", d1, d2, d3);

        for_each("copy_to_temporaries_output_tmp_0_ref",
            [&](int i, int j, int k) { copy_to_temporaries_output_tmp_0_ref(i, j, k) = 0.25 * in(i, j, k); });
        for_each("copy_to_temporaries_output_tmp_1_ref",
            [&](int i, int j, int k) { copy_to_temporaries_output_tmp_1_ref(i, j, k) = 0.75 * in(i, j, k); });

        ASSERT_TRUE(verify_storages(copy_to_temporaries_input_in, in, grid, halo_size));
        ASSERT_TRUE(verify_storages(copy_to_temporaries_output_in, in, grid, halo_size));
        ASSERT_TRUE(
            verify_storages(copy_to_temporaries_output_tmp_0, copy_to_temporaries_output_tmp_0_ref, grid, halo_size));
        ASSERT_TRUE(
            verify_storages(copy_to_temporaries_output_tmp_1, copy_to_temporaries_output_tmp_1_ref, grid, halo_size));

        // Verify: "copy_from_temporaries"
        storage_t &copy_from_temporaries_input_out_ref =
            make_storage("copy_from_temporaries_input_out_ref", d1, d2, d3);
        for_each("copy_from_temporaries_input_out_ref",
            [&](int i, int j, int k) { copy_from_temporaries_input_out_ref(i, j, k) = -1; });

        ASSERT_TRUE(
            verify_storages(copy_from_temporaries_input_out, copy_from_temporaries_input_out_ref, grid, halo_size));
        ASSERT_TRUE(
            verify_storages(copy_from_temporaries_input_tmp_0, copy_to_temporaries_output_tmp_0_ref, grid, halo_size));
        ASSERT_TRUE(
            verify_storages(copy_from_temporaries_input_tmp_1, copy_to_temporaries_output_tmp_1_ref, grid, halo_size));
        ASSERT_TRUE(verify_storages(copy_from_temporaries_output_out, out, grid, halo_size));
        ASSERT_TRUE(
            verify_storages(copy_from_temporaries_output_tmp_0, copy_to_temporaries_output_tmp_0_ref, grid, halo_size));
        ASSERT_TRUE(
            verify_storages(copy_from_temporaries_output_tmp_1, copy_to_temporaries_output_tmp_1_ref, grid, halo_size));

        // Rerun stencil
        // =============
        copy_with_temporaries->run(serializer, "copy_with_temporaries");
        copy_with_temporaries->finalize();

        ser::serializer ref_serializer2(ser::open_mode::Read, directory(), prefix());

        // Check global stencil list is still unqiue
        ASSERT_TRUE(ref_serializer2.global_meta_info().has_key("stencils"));
        ASSERT_EQ(ref_serializer2.get_global_meta_info_as< std::vector< std::string > >("stencils").size(), 1);
        ASSERT_EQ(ref_serializer2.get_global_meta_info_as< std::vector< std::string > >("stencils")[0],
            "copy_with_temporaries");

        ASSERT_EQ(ref_serializer2.savepoints().size(), 8);
        ASSERT_EQ(ref_serializer2.fieldnames().size(), 4);

    } catch (std::exception &e) {
        ASSERT_TRUE(false) << e.what();
    }
}

#endif
