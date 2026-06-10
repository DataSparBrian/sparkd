# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Startup no longer appears to hang. In-process Alembic migrations called
  `fileConfig()` with the default `disable_existing_loggers=True`, which disabled
  uvicorn's loggers and swallowed the `Application startup complete` /
  `Uvicorn running on …` lines — a healthy server looked dead. Migrations now
  preserve existing loggers (`disable_existing_loggers=False`).
