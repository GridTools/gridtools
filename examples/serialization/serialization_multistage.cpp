/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

struct copy_to_foo {
    typedef accessor< 0, enumtype::in, extent<>, 3 > in;
    typedef accessor< 1, enumtype::inout, extent<>, 3 > foo;

    typedef boost::mpl::vector< in, foo > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
        eval(foo()) = eval(in());
    }
};

struct copy_from_foo {
    typedef accessor< 0, enumtype::in, extent<>, 3 > foo;
    typedef accessor< 1, enumtype::inout, extent<>, 3 > out;

    typedef boost::mpl::vector< foo, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
        eval(out()) = eval(foo());
    }
};

TEST_F(serialization_setup, multistage) {
    namespace ser = serialbox::gridtools;
    uint_t d1 = 3, d2 = 4, d3 = 5;

    // Storages
    storage_t &in = make_storage("in", d1, d2, d3);
    storage_t &out = make_storage("out", d1, d2, d3);
    storage_t &foo = make_storage("foo", d1, d2, d3);

    auto in_view = make_host_view(in);
    auto out_view = make_host_view(out);
    auto foo_view = make_host_view(foo);

    for_each("in", [&](int i, int j, int k) { in_view(i, j, k) = i + j + k; });
    for_each("out", [&](int i, int j, int k) { out_view(i, j, k) = -1; });
    for_each("foo", [&](int i, int j, int k) { foo_view(i, j, k) = -1; });

    // Domain
    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef arg< 2, storage_t > p_foo;
    typedef boost::mpl::vector< p_in, p_out, p_foo > accessor_list;

    gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out), (p_foo() = foo));

    // Grid
    halo_descriptor di{0, 0, 0, d1 - 1, d1};
    halo_descriptor dj{0, 0, 0, d2 - 1, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    // Assemble stencil
    auto copy = make_computation< backend_t >(domain,
        grid,
        make_multistage(execute< forward >(), make_stage< copy_to_foo >(p_in(), p_foo())),
        make_multistage(execute< forward >(), make_stage< copy_from_foo >(p_foo(), p_out())));

    try {
        // Run & serialize
        // ===============
        ser::serializer serializer(ser::open_mode::Write, directory(), prefix());
        copy->ready();
        copy->steady();
        copy->run(serializer, "multistage");
        copy->finalize();

        // Verify
        // ======
        ser::serializer ref_serializer(ser::open_mode::Read, directory(), prefix());

        // Check functor
        ASSERT_TRUE(verify_storages(in, out, grid));

        // Check field meta-information
        ASSERT_TRUE(verify_field_meta_info(ref_serializer));

        // Check serialized data
        ASSERT_EQ(ref_serializer.savepoints().size(), 4);
        ASSERT_EQ(ref_serializer.fieldnames().size(), 3);
        ASSERT_TRUE(ref_serializer.global_meta_info().has_key("stencils"));
        ASSERT_EQ(ref_serializer.get_global_meta_info_as< std::vector< std::string > >("stencils")[0], "multistage");

        // Load serialized data
        // ====================

        // Savepoint: input of "copy_to_foo"
        storage_t &copy_to_foo_input_in = make_storage("copy_to_foo_input_in", d1, d2, d3);
        storage_t &copy_to_foo_input_foo = make_storage("copy_to_foo_input_foo", d1, d2, d3);
        ref_serializer.read("in", ref_serializer.savepoints()[0], copy_to_foo_input_in);
        ref_serializer.read("foo", ref_serializer.savepoints()[0], copy_to_foo_input_foo);

        // Savepoint: output of "copy_to_foo"
        storage_t &copy_to_foo_output_in = make_storage("copy_to_foo_output_in", d1, d2, d3);
        storage_t &copy_to_foo_output_foo = make_storage("copy_to_foo_output_foo", d1, d2, d3);
        ref_serializer.read("in", ref_serializer.savepoints()[1], copy_to_foo_output_in);
        ref_serializer.read("foo", ref_serializer.savepoints()[1], copy_to_foo_output_foo);

        // Savepoint: input of "copy_from_foo"
        storage_t &copy_from_foo_input_out = make_storage("copy_from_foo_input_out", d1, d2, d3);
        storage_t &copy_from_foo_input_foo = make_storage("copy_from_foo_input_foo", d1, d2, d3);
        ref_serializer.read("out", ref_serializer.savepoints()[2], copy_from_foo_input_out);
        ref_serializer.read("foo", ref_serializer.savepoints()[2], copy_from_foo_input_foo);

        // Savepoint: output of "copy_from_foo"
        storage_t &copy_from_foo_output_out = make_storage("copy_from_foo_output_out", d1, d2, d3);
        storage_t &copy_from_foo_output_foo = make_storage("copy_from_foo_output_foo", d1, d2, d3);
        ref_serializer.read("out", ref_serializer.savepoints()[3], copy_from_foo_output_out);
        ref_serializer.read("foo", ref_serializer.savepoints()[3], copy_from_foo_output_foo);

        // Verify serialized data
        // ======================
        storage_t &input_out_ref = make_storage("input_out_ref", d1, d2, d3);
        auto input_out_ref_view = make_host_view(input_out_ref);
        storage_t &input_foo_ref = make_storage("input_foo_ref", d1, d2, d3);
        auto input_foo_ref_view = make_host_view(input_foo_ref);
        for_each("input_out_ref", [&input_out_ref_view](int i, int j, int k) { input_out_ref_view(i, j, k) = -1; });
        for_each("input_foo_ref", [&input_foo_ref_view](int i, int j, int k) { input_foo_ref_view(i, j, k) = -1; });

        ASSERT_TRUE(verify_storages(copy_to_foo_input_in, in, grid));
        ASSERT_TRUE(verify_storages(copy_to_foo_input_foo, input_foo_ref, grid));
        ASSERT_TRUE(verify_storages(copy_to_foo_output_in, in, grid));
        ASSERT_TRUE(verify_storages(copy_to_foo_output_foo, foo, grid));
        ASSERT_TRUE(verify_storages(copy_from_foo_input_out, input_out_ref, grid));
        ASSERT_TRUE(verify_storages(copy_from_foo_input_foo, foo, grid));
        ASSERT_TRUE(verify_storages(copy_from_foo_output_out, out, grid));
        ASSERT_TRUE(verify_storages(copy_from_foo_output_foo, foo, grid));

    } catch (std::exception &e) {
        ASSERT_TRUE(false) << e.what();
    }
}

#endif
