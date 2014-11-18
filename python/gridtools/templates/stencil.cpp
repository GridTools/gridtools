#include "{{ stencil.hdr_file }}"


extern "C"
{
    int start (int x, int y, int z)
    {
        return !{{ stencil.name|lower }}::test(x, y, z);
    }
}
