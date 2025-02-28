# Vendoring Boost.Preprocessor

This directory contains a helper script for vendoring Boost.Preprocessor,
renaming the root directory, updating the includes to match the new root, and
updating macros to a new prefix. The script assumes we're dealing with
Boost.Preprocessor, which means no includes outside of Boost.Preprocessor, and
no namespace or other symbol renamings are required. The script transitively
copies only the required headers.

## First vendoring

When vendoring the first time, the first step requires finding direct
includes of Boost.Preprocessor. This can be done e.g. with:

``` shell
rg --only-matching --no-filename '\s*#\s*include\s+<(boost/preprocessor.*)>' --replace '$1' <path/to/gridtools> | sort -u
```

Note that using rg/ripgrep is useful since it will by default take .gitignore into account.

The first vendoring can be done with:

``` shell
<path/to/gridtools>/tools/vendor_boost_preprocessor/vendor_boost_preprocessor.py \
    --input-dir <path/to/boost>/libs/preprocessor/include/ \
    --output-dir <path/to/gridtools>/include/ \
    --macro-prefix GT \
    --include-prefix gridtools \
    --headers $(rg --only-matching --no-filename '\s*#\s*include\s+<(boost/preprocessor.*)>' --replace '$1' <path/to/gridtools> | sort -u)
```

Replace Boost includes with the new vendored includes:

``` shell
sed -i 's|<boost/|<gridtools/|g' <path/to/gridtools>/{tests,include}/**/*.*
```

The above assumes that Boost.Preprocessor is the only library used from Boost.

Replace Boost.Preprocessor macros:

``` shell
sed -i 's|BOOST_PP|GT_PP|g' {tests,include}/**/*.*
```

## Updating the vendored headers

If the vendored Boost.Preprocessor needs to be updated, or the required headers
change, the new list of required headers can be found with:

``` shell
rg --only-matching --no-filename --glob '!include/gridtools/preprocessor' '\s*#\s*include\s+<(gridtools/preprocessor.*)>' --replace '$1' | sort -u | sed 's|gridtools|boost|'
```

Compared to the original, this looks for `gridtools/preprocessor` includes,
instead of `boost/preprocessor`, since we've already vendored the library. It
also ignores the vendored library itself with `--glob`. Note that the ignore
pattern is relative to the current working directory, so it's easiest to run
this in the root of the repository. Since the vendoring script expects to find
headers under boost, the last `sed` renames `gridtools` to `boost`.

Using the above command to find the required headers, remove the vendored
Boost.Preprocessor directory, and repeat the vendoring command. Rename includes
and macros if needed.
