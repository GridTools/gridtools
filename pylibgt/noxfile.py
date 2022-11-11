import pathlib
import nox

nox.options.sessions = ["tests", "test_wheel"]


@nox.session
def tests(session):
    session.install(".")
    session.install("pytest")
    session.run("pytest", "tests", *session.posargs)


@nox.session
def test_wheel(session):
    session.install("pip")
    session.run("pip", "wheel", "-w", "dist", ".")
    wheel_path = list(pathlib.Path("./dist").glob("gridtools-*.whl"))[0]
    session.install(str(wheel_path))
    session.run("pytest", "tests", *session.posargs)
