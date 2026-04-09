"""
Test suite for project structure validation.

Tests check for file existence, readability, and non-zero sizes
for all source and test modules.
"""

import os

# Project root: one level up from tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestProjectStructure:
    """Tests for overall project structure."""

    def test_src_directory_has_files(self):
        """Source directory should have Python files."""
        src_dir = os.path.join(PROJECT_ROOT, "src")
        py_files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        assert len(py_files) > 0, "src directory should have Python files"

    def test_tests_directory_has_test_files(self):
        """Tests directory should have test files."""
        test_dir = os.path.join(PROJECT_ROOT, "tests")
        test_files = [f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]
        assert len(test_files) > 0, "tests directory should have test files"

    def test_tests_conftest_exists(self):
        """conftest.py should exist in tests directory."""
        conftest_path = os.path.join(PROJECT_ROOT, "tests", "conftest.py")
        assert os.path.exists(conftest_path), "conftest.py should exist"

    def test_tests_conftest_is_readable(self):
        """conftest.py should be readable."""
        conftest_path = os.path.join(PROJECT_ROOT, "tests", "conftest.py")
        assert os.access(conftest_path, os.R_OK), "conftest.py should be readable"


class TestSourceFiles:
    """Tests for source module files."""

    def test_creep_engine_exists(self):
        """creep_engine.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "creep_engine.py")
        assert os.path.exists(path)

    def test_oxidation_exists(self):
        """oxidation.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "oxidation.py")
        assert os.path.exists(path)

    def test_fatigue_exists(self):
        """fatigue.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "fatigue.py")
        assert os.path.exists(path)

    def test_creep_fatigue_exists(self):
        """creep_fatigue.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "creep_fatigue.py")
        assert os.path.exists(path)

    def test_tube_model_exists(self):
        """tube_model.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "tube_model.py")
        assert os.path.exists(path)

    def test_surrogate_exists(self):
        """surrogate.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "surrogate.py")
        assert os.path.exists(path)

    def test_monte_carlo_exists(self):
        """monte_carlo.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "monte_carlo.py")
        assert os.path.exists(path)

    def test_cybersecurity_exists(self):
        """cybersecurity.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "cybersecurity.py")
        assert os.path.exists(path)

    def test_config_exists(self):
        """config.py should exist."""
        path = os.path.join(PROJECT_ROOT, "src", "config.py")
        assert os.path.exists(path)


class TestTestFiles:
    """Tests for test file existence."""

    def test_test_creep_exists(self):
        """test_creep.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_creep.py")
        assert os.path.exists(path), "test_creep.py should exist"

    def test_test_oxidation_exists(self):
        """test_oxidation.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_oxidation.py")
        assert os.path.exists(path)

    def test_test_fatigue_exists(self):
        """test_fatigue.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_fatigue.py")
        assert os.path.exists(path)

    def test_test_creep_fatigue_exists(self):
        """test_creep_fatigue.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_creep_fatigue.py")
        assert os.path.exists(path)

    def test_test_tube_model_exists(self):
        """test_tube_model.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_tube_model.py")
        assert os.path.exists(path)

    def test_test_surrogate_exists(self):
        """test_surrogate.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_surrogate.py")
        assert os.path.exists(path)

    def test_test_monte_carlo_exists(self):
        """test_monte_carlo.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_monte_carlo.py")
        assert os.path.exists(path)

    def test_test_cybersecurity_exists(self):
        """test_cybersecurity.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_cybersecurity.py")
        assert os.path.exists(path)

    def test_test_visualization_exists(self):
        """test_visualization.py should exist."""
        path = os.path.join(PROJECT_ROOT, "tests", "test_visualization.py")
        assert os.path.exists(path)


class TestFileIsReadable:
    """Tests that key files are readable."""

    def test_all_test_files_readable(self):
        """All test files should be readable."""
        test_dir = os.path.join(PROJECT_ROOT, "tests")
        test_files = [f for f in os.listdir(test_dir)
                      if f.startswith("test_") and f.endswith(".py")]

        for test_file in test_files:
            path = os.path.join(test_dir, test_file)
            assert os.access(path, os.R_OK), f"{test_file} should be readable"

    def test_all_source_files_readable(self):
        """All source files should be readable."""
        src_dir = os.path.join(PROJECT_ROOT, "src")
        py_files = [f for f in os.listdir(src_dir) if f.endswith(".py")]

        for py_file in py_files:
            path = os.path.join(src_dir, py_file)
            assert os.access(path, os.R_OK), f"{py_file} should be readable"


class TestFileNonZeroSize:
    """Tests that key files are non-empty."""

    def test_test_files_have_content(self):
        """Test files should be non-zero size."""
        test_dir = os.path.join(PROJECT_ROOT, "tests")
        test_files = [f for f in os.listdir(test_dir)
                      if f.startswith("test_") and f.endswith(".py")]

        for test_file in test_files:
            path = os.path.join(test_dir, test_file)
            size = os.path.getsize(path)
            assert size > 0, f"{test_file} should be non-empty"

    def test_source_files_have_content(self):
        """Source files should be non-zero size."""
        src_dir = os.path.join(PROJECT_ROOT, "src")
        py_files = [f for f in os.listdir(src_dir) if f.endswith(".py")]

        for py_file in py_files:
            path = os.path.join(src_dir, py_file)
            size = os.path.getsize(path)
            assert size > 0, f"{py_file} should be non-empty"

    def test_conftest_has_content(self):
        """conftest.py should be non-empty."""
        conftest_path = os.path.join(PROJECT_ROOT, "tests", "conftest.py")
        size = os.path.getsize(conftest_path)
        assert size > 100, "conftest.py should have meaningful content"


class TestConfigurationFile:
    """Tests for configuration file."""

    def test_config_file_readable(self):
        """config.py should be readable."""
        config_path = os.path.join(PROJECT_ROOT, "src", "config.py")
        assert os.access(config_path, os.R_OK)

    def test_config_file_has_size(self):
        """config.py should have content."""
        config_path = os.path.join(PROJECT_ROOT, "src", "config.py")
        size = os.path.getsize(config_path)
        assert size > 1000, "config.py should have substantial content"


class TestAssetDirectory:
    """Tests for assets directory."""

    def test_assets_directory_exists(self):
        """Assets directory should exist."""
        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        assert os.path.isdir(assets_dir), "assets directory should exist"

    def test_assets_directory_readable(self):
        """Assets directory should be readable."""
        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        assert os.access(assets_dir, os.R_OK), "assets directory should be readable"


class TestGitIgnorePatterns:
    """Tests to ensure project structure allows testing."""

    def test_src_package_valid(self):
        """src should be a valid Python package."""
        init_path = os.path.join(PROJECT_ROOT, "src", "__init__.py")
        assert os.path.exists(init_path), "src/__init__.py should exist"

    def test_tests_can_import_src(self):
        """Tests should be able to import from src."""
        conftest_path = os.path.join(PROJECT_ROOT, "tests", "conftest.py")
        assert os.path.exists(conftest_path)

        with open(conftest_path, 'r') as f:
            content = f.read()
            assert "sys.path.insert" in content, "conftest should modify sys.path"
