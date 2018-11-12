# collect all tests in current folder
fetch_x86_tests(. LABELS unittest common)
fetch_mc_tests(. LABELS unittest common)
fetch_gpu_tests(. LABELS unittest common)

# collect all tests in generic_metafunctions folder
fetch_x86_tests(generic_metafunctions LABELS unittest common meta)
fetch_mc_tests(generic_metafunctions LABELS unittest common meta)
fetch_gpu_tests(generic_metafunctions LABELS unittest common meta)
