# collect test cases
fetch_x86_tests(. LABELS unittest storage)
fetch_mc_tests(. LABELS unittest storage)
fetch_gpu_tests(. LABELS unittest storage)

fetch_x86_tests(common LABELS unittest storage)
fetch_mc_tests(common LABELS unittest storage)
fetch_gpu_tests(common LABELS unittest storage)

fetch_x86_tests(storage_cuda LABELS unittest storage)
fetch_mc_tests(storage_cuda LABELS unittest storage)
fetch_gpu_tests(storage_cuda LABELS unittest storage)

fetch_x86_tests(storage_x86 LABELS unittest storage)
fetch_mc_tests(storage_x86 LABELS unittest storage)
fetch_gpu_tests(storage_x86 LABELS unittest storage)

fetch_x86_tests(storage_mc LABELS unittest storage)
fetch_mc_tests(storage_mc LABELS unittest storage)
fetch_gpu_tests(storage_mc LABELS unittest storage)
