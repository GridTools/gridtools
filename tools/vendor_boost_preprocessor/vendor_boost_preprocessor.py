#!/usr/bin/env python3

import os
import shutil
import argparse
import re


def extract_transitive_includes(header_path):
    includes = set()

    with open(header_path, "r", encoding="utf-8") as f:
        for line in f:
            # NOTE: This also has cover files included indirectly through macro
            # definitions, e.g.: #define PP1 <boost/preprocessor/iterate.hpp>,
            # so there's no "#include" in the pattern.
            match = re.search(r'[<"]?(boost/[^">]+)[">]', line)
            if match:
                includes.add(match.group(1))

    return includes


def find_headers(input_dir, initial_headers):
    all_headers = set(initial_headers)
    queue = list(initial_headers)

    while queue:
        current_header = queue.pop()
        current_path = os.path.join(input_dir, current_header)

        includes = extract_transitive_includes(current_path)

        for include in includes:
            if include not in all_headers:
                all_headers.add(include)
                queue.append(include)

    return all_headers


def vendor_header(input_path, output_path, macro_prefix, include_prefix):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace BOOST_* macros. NOTE: The pattern has to be able to match multiple
    # define BOOST_PP_NOT_EQUAL_CHECK_BOOST_PP_NOT_EQUAL_513(c, y) 0
    macro_prefix_upper = macro_prefix.upper()
    content = re.sub(r"BOOST_([A-Z0-9]+)", rf"{macro_prefix_upper}_\1", content)

    # Replace boost/* includes
    content = re.sub(r'(\s+[<"]?)boost/', rf"\1{include_prefix}/", content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def vendor_headers(input_dir, output_dir, headers, macro_prefix, include_prefix):
    os.makedirs(output_dir, exist_ok=True)

    for header in headers:
        # All includes are assumed to be relative to the Boost.Preprocessor
        # input directory
        input_path = os.path.join(input_dir, header)
        output_path = os.path.join(
            output_dir, header.replace("boost", include_prefix, 1)
        )

        # Ensure the output directory structure exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        vendor_header(input_path, output_path, macro_prefix, include_prefix)


def main():
    parser = argparse.ArgumentParser(description="Vendor Boost.Preprocessor.")

    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Path to the include directory of Boost.Preprocessor.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Path to the include directory where Boost.Preprocessor is vendored.",
    )
    parser.add_argument(
        "-e",
        "--headers",
        nargs="+",
        required=True,
        help="List of headers from Boost.Preprocessor to include. Transitive includes are detected from this list.",
    )
    parser.add_argument(
        "--macro-prefix",
        required=True,
        help="Prefix to use for replacing BOOST_PP in macros.",
    )
    parser.add_argument(
        "--include-prefix",
        required=True,
        help="Prefix to use for include directories of the vendored Boost.Preprocessor. Files will be copied to this prefix under the output directory, and includes changed to use this prefix.",
    )

    args = parser.parse_args()

    # Find all files in Boost.Preprocessor given the list of headers, including
    # transitive dependencies
    all_headers = find_headers(args.input_dir, set(args.headers))

    vendor_headers(
        args.input_dir,
        args.output_dir,
        all_headers,
        args.macro_prefix,
        args.include_prefix,
    )


if __name__ == "__main__":
    main()
