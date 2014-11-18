#include "{{ stencil.hdr_file }}"


extern "C"
{
    int run (int x, int y, int z)
    {
        return !{{ stencil.name|lower }}::test(x, y, z);
    }
}
