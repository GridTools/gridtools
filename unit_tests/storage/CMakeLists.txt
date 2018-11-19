# collect test cases
fetch_x86_tests(. LABELS unittest_x86)
fetch_mc_tests(. LABELS unittest_mc)
fetch_gpu_tests(. LABELS unittest_cuda)

fetch_x86_tests(common LABELS unittest_x86)
fetch_mc_tests(common LABELS unittest_mc)
fetch_gpu_tests(common LABELS unittest_cuda)

fetch_x86_tests(storage_cuda LABELS unittest_x86)
fetch_mc_tests(storage_cuda LABELS unittest_mc)
fetch_gpu_tests(storage_cuda LABELS unittest_cuda)

fetch_x86_tests(storage_x86 LABELS unittest_x86)
fetch_mc_tests(storage_x86 LABELS unittest_mc)
fetch_gpu_tests(storage_x86 LABELS unittest_cuda)

fetch_x86_tests(storage_mc LABELS unittest_x86)
fetch_mc_tests(storage_mc LABELS unittest_mc)
fetch_gpu_tests(storage_mc LABELS unittest_cuda)
