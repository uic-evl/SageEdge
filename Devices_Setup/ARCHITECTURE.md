# NIU Weather Dataset Stream Server - Architecture

This document outlines the architecture of the NIU Weather Dataset Stream Server, focusing on the modular design principles.

## Overview

The system has been redesigned with a modular architecture to improve maintainability, readability, and extensibility. The main functionality (streaming weather camera footage) has been split into logical components, each with clear responsibilities.

## Component Structure

### 1. Server Module (`server.py`)

The main entry point and orchestrator of the application:
- HTTP route handling
- Video streaming implementation
- User session management integration
- Request processing logic

### 2. Configuration Module (`config.py`)

Central location for all configuration settings:
- Environment detection (demo mode vs. real data)
- File paths and locations
- Streaming parameters
- Session management settings
- Frame rate options

### 3. Session Management (`session_manager.py`)

Handles user session state and cleanup:
- User session creation and retrieval
- Session state tracking
- Stream controller implementation
- Automatic session cleanup for inactive users

### 4. Image Processing (`image_processor.py`)

Responsible for all image and frame manipulation:
- Image loading from filesystem
- Demo frame generation
- Timestamp extraction
- Image batch processing
- Frame interpolation

### 5. API Routes (`api_routes.py`)

Defines supplemental API endpoints and is registered from `server.py` at startup:
- Status snapshots for clients (/api/status)
- Camera discovery (/api/cameras)
- Session info (/api/session)
- Health (/api/health)
- Analysis data endpoints (/api/analysis/*)

Note: The primary control plane for playback is `/control` (POST actions) exposed by `server.py`; `/api/*` covers read-only info plus some optional control helpers.

### 6. Utilities (`utils.py`)

Common utility functions used across modules:
- Frame encoding
- Timestamp formatting and parsing
- Frame enhancement (overlays, annotations)
- Rate limiting for smooth playback

## Data Flow

1. User requests connect to `server.py`
2. Session is created/retrieved via `session_manager.py`
3. Image data is processed through `image_processor.py`
4. Configuration from `config.py` dictates behavior
5. API requests are handled by registered routes in `api_routes.py`
6. Common operations use functions from `utils.py`

## Advantages of Modular Architecture

- **Separation of concerns**: Each module has a specific responsibility
- **Better maintainability**: Easier to locate and fix bugs
- **Code reusability**: Common functions are centralized
- **Easier testing**: Modules can be tested in isolation
- **Scalability**: New features can be added with minimal disruption
- **Team collaboration**: Different team members can work on different modules

## Future Extensibility

The modular design allows for easy extension:
- Additional image processing algorithms
- New API endpoints
- Alternative data sources
- Enhanced session management
- Authentication integration
