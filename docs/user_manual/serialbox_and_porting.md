#SerialBox and porting your reference application

 I'm here and I'm loving every minute of it
 

[^1]: With the Cuda backend we allocate memory on host and device. In
    the standard use-cases you don’t need to update the data manually,
    but you still have the option to do so.

[^2]: [``]{} is not yet supported.

[^3]: At this point the reader should be able to complete the missing
    parts in the setup.

[^4]: There are other ways to accomplish this behavior. The extra
    computation can be avoided by defining the Laplacian only on the
    interval where we need it; the temporary could be avoided by a bit
    of code duplication, however there is no good reason to do it.
