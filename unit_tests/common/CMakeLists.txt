# collect all tests in current folder
fetch_x86_tests(. LABELS unittest_x86 )
fetch_mc_tests(. LABELS unittest_mc )
fetch_gpu_tests(. LABELS unittest_cuda )

# collect all tests in generic_metafunctions folder
fetch_x86_tests(generic_metafunctions LABELS unittest_x86 )
fetch_mc_tests(generic_metafunctions LABELS unittest_mc )
fetch_gpu_tests(generic_metafunctions LABELS unittest_cuda )
