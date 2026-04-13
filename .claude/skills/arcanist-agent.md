# Skill: Using the Arcanist Agent

## When to Use

Load this skill when you need to understand how to interact with the Arcanist agent, what it can do, and the exact methods for getting the best results from it. This covers task delegation, available capabilities, communication patterns, and workflow conventions.

---

## What Is Arcanist

Arcanist is an AI background coding agent that runs inside a sandboxed environment against your repository. It reads your codebase, makes changes, runs commands, and commits results -- all without requiring manual intervention after you submit a task. It operates asynchronously: you give it a task, it executes, and you review the output (typically a commit or PR).

Arcanist is powered by Claude (`claude-opus-4-6`). It follows instructions from `CLAUDE.md` at the repository root and any skill files in `.claude/skills/`.

---

## How to Give Arcanist a Task

### Effective Task Format

Be specific and declarative. State what you want done, not how to navigate the UI.

**Good examples:**

```
Add a retry mechanism to the API client in src/client.py with exponential backoff. 
Max 3 retries, starting at 1 second delay. Add tests.
```

```
Fix the bug where user sessions expire after 5 minutes instead of 30. 
The timeout config is in src/config.py.
```

```
Refactor the payment module to separate Stripe logic from the core billing 
interface. Keep all existing tests passing.
```

**Poor examples:**

```
Make the code better.
```

```
Something is broken, can you look?
```

### What to Include in a Task

| Element | Why It Matters |
|---------|---------------|
| The specific change or fix needed | Arcanist needs a clear objective |
| Relevant file paths (if known) | Reduces investigation time |
| Expected behavior after the change | Defines the success criteria |
| Whether to run tests after | Arcanist will verify if asked |
| Whether to commit | Arcanist commits by default when done |

---

## Arcanist's Capabilities

### File Operations

Arcanist can read, write, edit, and search files across your entire repository.

| Operation | Method | Notes |
|-----------|--------|-------|
| Read a file | `Read` tool | Reads with line numbers, supports offset/limit |
| Write a file | `Write` tool | Overwrites entire file; always reads first |
| Edit a file | `Edit` tool | Exact string replacement; reads file first |
| Find files by name | `Glob` tool | Supports patterns like `**/*.py`, `src/**/*.ts` |
| Search file contents | `Grep` tool | Regex search across the codebase |

Arcanist always reads a file before editing it. It never guesses at file contents.

### Terminal Commands

Arcanist runs shell commands via `Bash` in a persistent Linux session. Common uses:

- Running tests (`pytest`, `npm test`, `go test`)
- Installing dependencies (`uv sync`, `npm install`)
- Running linters and formatters (`ruff check .`, `black .`)
- Git operations (status, diff, log, add, commit)
- Build commands (`npm run build`, `make`)
- Any CLI tool available in the environment

### Git Operations

Arcanist follows strict git safety rules:

| Action | Behavior |
|--------|----------|
| **Commits** | Creates commits automatically when work is done |
| **Staging** | Always stages specific files by name (never `git add .` or `git add -A`) |
| **Branches** | Does not create branches or push -- the sandbox handles this automatically |
| **Amend** | Never amends unless the commit was just created and not pushed, and only to include auto-formatted files |
| **Force push** | Never runs force push or destructive git commands |
| **Hooks** | Never skips pre-commit hooks |

### Codebase Exploration

Arcanist can delegate exploration to sub-agents for parallel search:

| Sub-agent Type | Purpose |
|----------------|---------|
| `explore` | Fast, read-only codebase search (find files, grep for patterns, answer questions about code) |
| `general` | Multi-step work that requires reading, reasoning, and potentially writing |

Arcanist launches multiple explore tasks in parallel when investigating independent questions.

### External Integrations

| Integration | What It Does |
|-------------|-------------|
| **GitHub** (`gh` CLI) | View PRs, issues, checks, releases. Never uses WebFetch for GitHub URLs. |
| **Web Search** | Searches the web for current information via Exa AI |
| **Code Search** | Finds code examples, documentation, and API references for libraries |
| **Context7** | Queries up-to-date library documentation (resolve library ID first, then query) |
| **Slack** | Sends messages, reads channels/threads, searches message history, adds reactions |
| **Braintrust** | Queries logs, lists projects/experiments, logs evaluation data |

### Task Management

For complex, multi-step work, Arcanist uses a structured todo list:

- Breaks work into discrete, trackable items
- Marks tasks `pending`, `in_progress`, `completed`, or `cancelled`
- Only one task is `in_progress` at a time
- Updates status in real-time as work progresses

### Asking Clarifying Questions

When Arcanist encounters ambiguity, it can ask you questions with selectable options. It only asks when:

- It needs user-specific information it cannot find in the repo
- A decision is high-stakes or irreversible
- The repo has no default or convention to follow

Otherwise, it decides autonomously and states: "Decided: [choice]. [reason]."

---

## How Arcanist Works on a Task

### Typical Execution Flow

1. **Reads `CLAUDE.md`** and any linked documentation for project conventions
2. **Investigates** the codebase (searches, reads relevant files) -- spends at most 15% of its budget here
3. **Plans** the change using TodoWrite for multi-step tasks
4. **Re-reads the original task** to confirm the plan addresses it before writing code
5. **Implements** the change by editing existing files (prefers edits over new files)
6. **Verifies** by running tests, type checks, or linters as appropriate
7. **Commits** the change with a descriptive message
8. The sandbox automatically handles branch creation, push, and PR management

### Rules Arcanist Follows

- **Reads before editing.** Every file is read before modification.
- **Searches before creating.** Checks for existing utilities before writing new ones.
- **Fixes all callers.** When changing a function signature, updates every call site in the same commit.
- **Preserves valid state.** Error paths never destroy previously valid data.
- **No emojis in code/docs** unless you explicitly request them.
- **No speculation.** If a fact is verifiable, Arcanist verifies it with a tool call rather than guessing.
- **Honest reporting.** Never claims "all tests pass" unless the current run proves it.

---

## Configuring Arcanist for Your Repository

### CLAUDE.md

Place a `CLAUDE.md` file at your repository root. Arcanist reads this before every task. Include:

- Project overview and architecture
- Development principles and conventions
- Tooling commands (how to install, test, lint, build)
- Documentation standards
- File/directory descriptions
- Any rules specific to your project

### Skill Files

Place `.md` files in `.claude/skills/` to give Arcanist domain-specific knowledge. Each skill file should include:

- **When to Use** section -- describes when the skill is relevant
- Step-by-step workflows for specific tasks
- Code examples with exact API signatures
- Common patterns and pitfalls

Arcanist loads skills on demand when a task matches the skill's description.

### .env and Secrets

Arcanist respects `.env` files for environment variables. It will never commit files that likely contain secrets (`.env`, `credentials.json`, etc.) and warns you if you ask it to.

---

## Best Practices for Working with Arcanist

### Do

- **Be specific** about what you want changed and where
- **Mention file paths** if you know them -- saves investigation time
- **Ask for tests** explicitly if you want them written or run
- **Reference issues or PRs** by number -- Arcanist can look them up with `gh`
- **State success criteria** so Arcanist knows when the task is complete
- **Keep CLAUDE.md updated** -- it is the single source of truth for project conventions

### Do Not

- **Do not ask vague questions** without context ("fix the bug", "make it faster")
- **Do not expect GUI interaction** -- Arcanist runs in a headless terminal
- **Do not assume it remembers prior sessions** -- each task starts fresh
- **Do not ask it to push or create PRs** -- the sandbox handles this automatically
- **Do not include secrets** in your task descriptions

---

## Example Task Prompts

### Bug Fix

```
The `calculate_total` function in nons/core/scheduler.py returns incorrect values 
when `requests_per_minute` is set to 0. It should raise a ConfigurationError instead 
of dividing by zero. Add a test for this edge case.
```

### New Feature

```
Add a `timeout` parameter to the Agent class constructor in nons/core/agents/agent.py. 
It should set a global timeout (in seconds) for the entire agent.run() loop. Default to 
300 seconds. If exceeded, the generator should yield a timeout error and stop. Update 
the existing tests in tests/test_integration.py.
```

### Refactoring

```
Extract the rate limiting logic from nons/core/scheduler.py into a separate 
nons/core/rate_limiter.py module. Keep the public API of RequestScheduler unchanged. 
Ensure all existing tests still pass.
```

### Code Review Follow-up

```
Address the review comments on PR #42. The reviewer asked to:
1. Add docstrings to all public methods in nons/operators/base.py
2. Replace the bare except clause in nons/utils/providers.py line 87 with specific exceptions
3. Add type hints to the return values in nons/core/config.py
```
