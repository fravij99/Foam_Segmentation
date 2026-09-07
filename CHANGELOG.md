# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-07

### Added
- **Python Package Structure:** Refactored the core library into an installable `foam_segmentation` package (via `pyproject.toml`).
- **Web Dashboard:** Added a Flask backend (`app.py`) to serve a modern glassmorphism web interface.
- **Local File Browser:** Implemented a GUI for users to browse and select their local image folders directly from the web browser.
- **Canvas ROI Selector:** Added an HTML5 Canvas-based ROI selector to replace Matplotlib blocking pop-ups.
- **Automated Tests:** Created comprehensive `pytest` coverage in `tests/test_segmentation.py` with 95% line coverage.
- **CI/CD Pipeline:** Added GitHub Actions workflow to run automated tests on push and pull requests.
- **Examples:** Moved script files (`bubbles.py`, `detecting_roi.py`, `foam_heigth.py`) into the `examples/` directory.

### Changed
- Refactored `src/foamlib.py` into `src/foam_segmentation/core.py`.
- Configured Matplotlib to use the `Agg` non-interactive backend to ensure thread-safety and avoid GUI crashes in headless environments.
- Updated Matplotlib saving functions to dynamically manage output folders based on inputs.

### Fixed
- Fixed `RectangleSelector` missing arguments for Matplotlib >= 3.5 compatibility.
- Handled `curve_fit` `RuntimeError` gracefully in `foam_progression_plot` to avoid crashes on identical image sets.
