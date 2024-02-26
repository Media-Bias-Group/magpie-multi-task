"""Use this file to define reusable test objects."""

from pytest import fixture 

# TODO: (jw) I used those lines in a different project; we may want/ need to use them sometime later.
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)


@fixture(scope="function")
def test_data():
    """Yield test data (String)."""
    data = "Some initial test data"
    yield data


@fixture(scope="function")
def test_blob_name():
    """Yield a test blob name."""
    blob_name = "test_blob_name"
    yield blob_name


@fixture(scope="function")
def test_filename():
    """Yield a test filename."""
    test_filename = "test_filename"
    return test_filename
