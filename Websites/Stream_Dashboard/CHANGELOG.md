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
