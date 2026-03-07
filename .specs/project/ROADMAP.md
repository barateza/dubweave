# Roadmap

**Current Milestone:** Production Readiness
**Status:** Planning

---

## Milestone 1: Production Readiness

**Goal:** Address all gaps identified in the production readiness assessment so Dubweave can be published as a reliable open-source tool.
**Target:** All 6 features verified and merged.

### Features

**F1: Documentation & Setup** — PLANNED

- Comprehensive installation guide for all platforms
- Environment setup with troubleshooting
- System requirements and dependency documentation
- Usage guide with examples

**F2: Error Handling & UX** — PLANNED

- User-friendly error messages for all failure modes
- Network failure and API rate limit handling
- Progress indicators for long-running operations

**F3: Security & Configuration** — PLANNED

- API key validation at startup
- URL and input sanitization hardening
- Secure temp file handling and cleanup

**F4: Performance & Scalability** — PLANNED

- Memory usage optimization for long videos
- Resource cleanup and disk space management
- GPU memory release between pipeline stages

**F5: Testing & Quality Assurance** — PLANNED

- Unit tests for core pure functions
- Integration tests for pipeline stages
- Edge case coverage (long videos, bad audio, missing segments)

**F6: Deployment & Monitoring** — PLANNED

- Structured logging (replace print-based `log()`)
- Health checks and startup validation
- Production configuration guide

---

## Future Considerations

- Module extraction (split app.py into packages)
- Linux/macOS support
- Docker containerization
- Multi-target-language support
- CI/CD with GitHub Actions
- Public Gradio.live deployment guide
