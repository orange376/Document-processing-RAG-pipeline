---
name: codegraph
description: >
  Index / build a code knowledge graph of a TypeScript/JavaScript repo, then query it
  to navigate relationships without grepping blindly. Use this whenever: the user asks
  to "index", "analyze", "map", or "understand" the codebase; you need to know what a
  change breaks; you want to find who calls/uses/renders a symbol; you're orienting in
  an unfamiliar repo before opening files; you need to understand what a PR diff
  impacts; or you need to find a symbol by description rather than exact name.
  The MCP server (inspect_symbol, analyze_impact, trace_path, find_dead_code,
  survey_repo, open_explorer, search_symbols, analyze_diff) builds and caches the graph
  automatically — no manual build step needed. The CLI (codegraph <source>) builds it
  explicitly to a graph.json for offline or agent-less use.
---

# codegraph — code knowledge graph for agents

A knowledge graph of the codebase: nodes (files, functions, classes, interfaces,
methods) connected by typed edges resolved by the **TypeScript type-checker** — so
relationships are **exact**, not guessed by name.

---

## Reach for this proactively — don't wait to be asked

Use codegraph **without the user explicitly requesting it** in these situations:

- **Starting work on an unfamiliar repo or feature area** → `survey_repo()` first.
  Costs one tool call, saves you from opening 10 wrong files.
- **About to change, refactor, or delete a symbol** → `analyze_impact()` before
  touching anything. You need to know the blast radius.
- **Navigating to understand how something works** → `inspect_symbol()` before grep.
  You get location + source + all relationships in one call.
- **User says "index this repo / build the graph / analyze the codebase"** → if the
  MCP is available, call `survey_repo()` (the graph builds automatically). If no MCP,
  run `codegraph . --out ./graph` and read `GRAPH_INDEX.md`.
- **Reviewing or about to merge a PR** → `analyze_diff()` to see what the diff
  actually impacts beyond the changed lines.
- **Looking for something but don't know its exact name** → `search_symbols()`.

**Golden rule:** For relationship questions that span files, query the graph BEFORE
opening files or grepping. Use grep/Read only to read the contents of what the graph
already pointed you to.

---

## Two modes: MCP (auto) vs CLI (manual)

### MCP tools — preferred, zero setup per repo

If the `codegraph` MCP server is available in your session, use these tools directly.
**They build and cache the graph automatically on first call** — no `codegraph build`
step needed. The graph is stored in `~/.cache/codegraph/<hash>/` (never pollutes the
repo) and rebuilt incrementally on subsequent calls.

| Tool | When to reach for it |
|------|----------------------|
| `survey_repo()` | First move in any unfamiliar repo. Also answers "index/map/analyze this repo". |
| `inspect_symbol(query)` | "What is X?", "show me X", "who calls X?". Default first tool for any symbol question. |
| `analyze_impact(symbol, depth?)` | Before changing/refactoring/deleting anything. `depth>1` for the transitive chain. |
| `trace_path(from, to)` | "How does A reach B?", "is there a link between X and Y?" |
| `find_dead_code()` | "What can I safely delete?", "what's unused?" |
| `search_symbols(query, limit?)` | Know what it does but not the name — "find the rate limiting logic". Auto-builds embeddings on first call. |
| `analyze_diff(base?, head?, depth?)` | "What does this PR/commit break?". Default range: `HEAD~1..HEAD`. |
| `open_explorer()` | User wants to **see** the graph visually in a browser — not for you, for the human. |

### CLI — when MCP is not available

Build the graph from the repo root, then query it. By default the output goes to
`~/.cache/codegraph/<hash>/` (keyed to the repo path) so the repo stays clean —
no `.gitignore` entry needed. Use `--out <dir>` only if you want the files somewhere
specific (e.g. to share them with the team).

```bash
# Build (run from repo root)
codegraph .                        # output → ~/.cache/codegraph/<hash>/
codegraph . --with-embeddings      # also build semantic embeddings (enables search)
codegraph . --open                 # build and open the visual explorer

# Queries — pass the graph.json path (find it in ~/.cache/codegraph/<hash>/graph.json
# or use --out ./graph to put it somewhere known)
# All queries accept --json for parseable output
codegraph query <graph.json> impact <symbol>              # what breaks?
codegraph query <graph.json> impact <symbol> --depth 3   # transitive blast radius
codegraph query <graph.json> neighbors <symbol>           # what X uses and who uses X
codegraph query <graph.json> path <A> <B>                 # shortest connection
codegraph query <graph.json> unused                       # dead-code candidates
codegraph query <graph.json> diff                         # last commit impact
codegraph query <graph.json> diff --base main --head HEAD # branch vs main
codegraph query <graph.json> search "rate limiting logic" # semantic search
codegraph mermaid <graph.json> <symbol> --depth 2         # focused diagram
```

---

## Keeping the graph fresh — when to regenerate

**The MCP server rebuilds incrementally on demand** — after each call it checks for
file changes and re-extracts only the modified files. You never need to trigger it
manually. Skip this section if you're using the MCP.

For the **CLI**, the graph is a snapshot. It goes stale when code changes. Regenerate when:

| Signal | Action |
|--------|--------|
| `meta.dirty: true` in `graph.json` | Working tree has uncommitted changes. Regenerate before making irreversible decisions (deleting, changing signatures). |
| `meta.commit` is behind `git log --oneline -1` | Code changed since the graph was built. Regenerate: `codegraph . --out ./graph` |
| You added/deleted/moved files | Regenerate. New files won't appear; deleted ones leave ghost nodes. |
| You refactored a module boundary or renamed exports | Regenerate. Name-based edges may be wrong. |
| Graph is more than a few hours old in an active repo | Regenerate to be safe. Incremental builds are cheap — the second run only re-extracts changed files. |

```bash
# Regenerate (incremental — only re-extracts changed files, fast)
codegraph .

# Force full rebuild (if the cache itself seems wrong)
codegraph . --no-cache
```

After regenerating, re-run whatever query you were about to make. Never act on a
stale graph for a destructive operation (delete, refactor, signature change).

---

## When NOT to use it

- **Single file already open**: just read it — graph adds overhead with no benefit.
- **Business logic, strings, config**: normal grep/Read. The graph is about
  relationships, not contents.
- **Already know the exact location**: go directly. The graph is a navigation aid
  when you _don't_ know where to look.

---

## MCP — enable/disable (only when user asks)

```bash
codegraph status   # is it currently registered?
codegraph enable   # register the MCP in Claude Code (user-scoped)
codegraph disable  # unregister it (skill stays installed)
```

**Only do this when the user explicitly asks.** It modifies their Claude Code config.
**Takes effect on the next session restart** — Claude Code loads MCP servers at startup.
If the tools are already visible in this session, the server is already running; don't
re-enable it.

---

## Disambiguation

When a name matches multiple symbols, the result shows a `⚠` with their ids. Use the
exact id `path/to/file.ts::name` to target the one you want.

---

## Edge type reference

| Edge | Meaning |
|------|---------|
| `calls` | symbol invokes another |
| `renders` | component uses another via JSX `<Comp/>` |
| `imports` | file imports from another (incl. barrel re-exports — `export … from`) |
| `extends` | class inheritance |
| `implements` | interface implementation |
| `references` | `new X`, function/component passed by reference (callback, `onClick={save}`), enum as value |
| `contains` | file declares a symbol |

---

## Typical flows

**Unfamiliar repo / "index this codebase"**
→ `survey_repo()` (MCP) — or `codegraph .` + read `GRAPH_INDEX.md` (CLI)
→ `inspect_symbol()` on the god nodes
→ open only the files that matter

**Before changing a symbol**
→ `analyze_impact(symbol)` — know the blast radius first
→ if depth=1 has many dependents, run `analyze_impact(symbol, 3)` for the full chain
→ open the specific dependents that need updating

**PR review**
→ `analyze_diff()` to map the diff to impacted symbols
→ `inspect_symbol()` on each changed symbol to see what uses it
→ decide scope of review based on actual impact, not just changed lines

**"Find the thing that does X"**
→ `search_symbols("X")` if embeddings exist
→ otherwise `inspect_symbol("likely name")` and follow edges
