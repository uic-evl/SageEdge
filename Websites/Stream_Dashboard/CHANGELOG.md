# Changelog

All notable changes to the NIU Weather Dataset Stream Server will be documented in this file.

## [1.0.0] - 2025-08-05

### Added
- Complete modular architecture refactoring
- Multi-user session support with automatic cleanup
- API endpoints for analysis data and camera control
- Demo mode for running without the NIU dataset
- Improved error handling and logging
- Detailed documentation in README.md and ARCHITECTURE.md

### Changed
- Separated code into logical modules:
  - config.py: Configuration settings
  - session_manager.py: User session management
  - image_processor.py: Image loading and processing
  - api_routes.py: API endpoint definitions
  - utils.py: Utility functions
  - server.py: Main server module
- Enhanced buffering system for smoother playback
- Improved timestamp extraction and handling
- Better frame interpolation for smooth transitions

### Fixed
- Memory leaks in long-running sessions
- Race conditions in multi-user environments
- Buffering issues when changing playback speed
- Error handling for missing or corrupted images

## [1.1.0] - 2025-08-12

### Added
- /status now returns `data_range` and `excluded_dates` for dynamic UI constraints.
- Client UI shows the actual time used when the requested time has no photos, and syncs inputs to it.

### Changed
- StreamController falls back to the nearest day with images when a requested window/day is empty.
- Frontend date picker now reads min/max/excluded dates from /status.

### Fixed
- Eliminated 500s from /status by importing missing config; corrected image listing edge case in producer.

## [1.2.0] - 2025-09-02

### Added
- Unified status builder module (`status_builder.py`) used by /status, /api/session, and deprecated /api/health.
- In-process rate limiting (session creation 10/min, control actions 60/min).

### Changed
- /api/session now returns the full unified status payload plus feed URLs.
- /status simplified to delegate to builder; consistent keys across endpoints.
- /api/health marked deprecated (returns { status, session }).
- FPS options trimmed to a max of 8fps (higher unused values removed).

### Removed
- Unused MIN_BUFFER constant and unused controller tracking fields.

### Fixed
- Eliminated redundant status construction logic and reduced risk of field drift across endpoints.
