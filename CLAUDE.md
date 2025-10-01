
# NoN – Technical Architecture

Welcome to the NoN (Network of Networks) package! This document provides an overview of the technical architecture, development standards, and documentation structure for this repository. Please read through carefully to understand the design principles, tooling, and documentation conventions.

---

## 1. Overview

NoN is designed as a modular, composable, and externally facing package for network orchestration and management. The architecture is built to be robust, maintainable, and easy to extend, following SOLID principles throughout.

---

## 2. Development Principles

- **SOLID Principles**: All code, especially functions and classes, must be well-documented and adhere to SOLID design principles to ensure maintainability and extensibility.
- **Predefined Functions**: When implementing new features, always search for existing, predefined functions before writing new code to avoid duplication.
- **Circular Dependency Checks**: Before merging or introducing new modules, check for circular dependencies to maintain a clean and reliable codebase.
- **Composability**: Structure code so that components are easily reusable and composable.

---

## 3. Tooling & Package Management

- **Package Management**: We use [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management. All dependencies are managed via `pyproject.toml` and `uv`.
- **Code Formatting**: All code must be formatted using [`black`](https://black.readthedocs.io/en/stable/) for consistency.
- **Environment**: Update the `.env` file as needed for environment variables. Ensure all required variables are documented in the root README.

---

## 4. Documentation Structure

### 4.1. READMEs

- Every directory **must** contain a `README.md` that explains the purpose of the directory and its contents.
- Each file should have a doc string at the top describing its purpose.
- Functions and classes must be documented with clear doc strings, including parameter and return type information.

### 4.2. File/Folder Documentation

| Path                | Description                                                      |
|---------------------|------------------------------------------------------------------|
| `/src`              | Main source code for NoN.                                        |
| `/src/README.md`    | Overview of the source code structure and modules.               |
| `/tests`            | Test suite for the package.                                      |
| `/tests/README.md`  | How to run and write tests.                                      |
| `/scripts`          | Utility and management scripts.                                  |
| `/scripts/README.md`| Documentation for each script and its usage.                     |
| `/docs`             | Additional documentation, architecture diagrams, etc.            |
| `/docs/README.md`   | Index of documentation resources.                                |
| `.env`              | Environment variables (documented in root README).               |
| `pyproject.toml`    | Project configuration and dependencies.                          |
| `Makefile`          | Common development commands.                                     |

---

## 5. How to Contribute

1. **Check for Predefined Functions**: Before implementing, search the codebase for existing utilities.
2. **Check for Circular Dependencies**: Use tools or scripts provided in `/scripts` to check for circular imports.
3. **Follow Documentation Standards**: Update or add `README.md` files as needed.
4. **Format Code**: Run `prettier` before committing.
5. **Install Dependencies**: Use `uv` for all package management tasks.
6. **Running Python Code**: Use `uv run` for running a Python file.

---

## 6. Additional Notes
- Don't use emojis anywhere in documentation in any circumstances. Keep the tone here serious and explain any concept using a vocabulary or additional break down section so a slightly technical PM could also read this effectively.

