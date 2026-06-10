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
