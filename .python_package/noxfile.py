# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import configparser
import pathlib
import shutil

import nox

nox.options.sessions = ["test_src_dist", "test_wheel"]


def prepare(session: nox.Session):
    session.install("cmake>=2.18.1")
    session.install("ninja")
    build_path = session.cache_dir.joinpath("build").absolute()
    build_path.mkdir(exist_ok=True)
    install_path = pathlib.Path(".").absolute() / "py_src" / "gridtools" / "data"
    source_path = pathlib.Path("..").absolute()
    with session.chdir(build_path):
        session.run(
            "cmake",
            "-DBUILD_TESTING=OFF",
            "-DGT_INSTALL_EXAMPLES:BOOL=OFF",
            f"-DCMAKE_INSTALL_PREFIX={install_path}",
            str(source_path),
        )
        session.run("make", "install")
        session.log("installed gridttols sources")
    version_path = source_path / "version.txt"
    setup_path = pathlib.Path(".") / "setup.cfg"
    config = configparser.ConfigParser()
    config.read(str(setup_path))
    config["metadata"]["version"] = version_path.read_text()
    with setup_path.open("w") as setup_fp:
        config.write(setup_fp)


def get_wheel(session: nox.Session) -> pathlib.Path:
    return list(session.cache_dir.joinpath("dist").glob("gridtools_cpp-*.whl"))[0]


@nox.session
def build_wheel(session: nox.Session):
    prepare(session)
    dist_path = session.cache_dir.joinpath("dist").absolute()
    workdir = pathlib.Path(".").absolute()
    session.install("build[virtualenv]")
    with session.chdir(session.cache_dir):
        session.run(
            "python",
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "-o",
            str(dist_path),
            str(workdir),
        )
    session.log(f"built wheel in {dist_path}")
    session.log("\n".join(str(path) for path in dist_path.iterdir()))


@nox.session
def test_src_dist(session: nox.Session):
    prepare(session)
    session.install(".")
    session.install("pytest")
    session.run("pytest", "tests", *session.posargs)


@nox.session
def test_wheel(session: nox.Session):
    session.notify("build_wheel")
    session.notify("test_wheel_with_python-3.8")
    session.notify("test_wheel_with_python-3.9")
    session.notify("test_wheel_with_python-3.10")
    session.notify("test_wheel_with_python-3.11")


@nox.session(python=["3.8", "3.9", "3.10", "3.11"])
def test_wheel_with_python(session: nox.Session):
    wheel_path = get_wheel(session)
    session.install("pytest")
    session.install(str(wheel_path))
    session.run("pytest", "tests", *session.posargs)


@nox.session
def clean_cache(session: nox.Session):
    for subtree in session.cache_dir.iterdir():
        shutil.rmtree(subtree, True)


@nox.session
def build(session: nox.Session):
    prepare(session)
    session.install("build[virtualenv]")
    session.run("python", "-m", "build", "--no-isolation", *session.posargs)


@nox.session
def clean(session: nox.Session):
    top_dir = pathlib.Path(".")
    data_dir = top_dir / "py_src" / "gridtools" / "data"
    session.log(f"rm -r {data_dir}")
    shutil.rmtree(data_dir, True)
    session.log("rm -r py_src/*.egg-info")
    for egg_tree in top_dir.joinpath("py_src").glob("*.egg-info"):
        shutil.rmtree(egg_tree, True)
    session.log("rm -r dist")
    shutil.rmtree(top_dir / "dist", True)
    session.log("rm -r build")
    shutil.rmtree(top_dir / "build", True)
