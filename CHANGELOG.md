# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Documentation
- Quick Start now documents the required frontend build step
  (`npm install && npm run build`). A fresh clone omits the git-ignored
  `sparkd/static/` bundle, so `/` returned `{"detail": "Not Found"}` until the
  UI was built. Also documents the Vite dev-server workflow and clarifies that
  the API runs without the UI.
### Fixed
- Startup no longer appears to hang. In-process Alembic migrations called
  `fileConfig()` with the default `disable_existing_loggers=True`, which disabled
  uvicorn's loggers and swallowed the `Application startup complete` /
  `Uvicorn running on …` lines — a healthy server looked dead. Migrations now
  preserve existing loggers (`disable_existing_loggers=False`).
