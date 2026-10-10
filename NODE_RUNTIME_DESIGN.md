# Moving PythonInterpreter from Deno to Node.js

## Summary

`dspy.PythonInterpreter` runs model-generated Python in Pyodide, which is CPython compiled to WebAssembly. Pyodide needs a JavaScript runtime to host it. DSPy uses Deno for that today. This branch replaces Deno with Node.js.

The change is small and contained. One Python module and one JavaScript runner change. No public module API changes except the removal of `deno_command`. The full interpreter, RLM, ProgramOfThought, CodeAct, and Flex test suites pass on Node.js 24 and on Node.js 26.

The main cost is in the security story. Deno's permission system was the sandbox boundary. The Node.js documentation says its permission model is not a security boundary against malicious code. This branch adds a second layer to make up for that, but reviewers should decide whether that layer is enough.

## How DSPy uses Deno today

Only `dspy/primitives/python_interpreter.py` and its runner script touch Deno. Every other module reaches Deno through `PythonInterpreter`.

These are the Deno features DSPy uses:

- Deno fetches Pyodide on first run through an `npm:pyodide@0.29.4` import and caches it.
- Deno flags `--allow-read`, `--allow-write`, `--allow-env`, and `--allow-net` limit what the process can touch. The `--allow-net` flag accepts a list of hosts.
- The runner calls `Deno.permissions.revoke` after start up so sandboxed code cannot read the shared Deno cache.
- The optional `dspy[deno]` extra installs the `deno` wheel from PyPI, so users get a pinned binary through pip.

The Python process talks to the runner over stdin and stdout with JSON-RPC. That protocol does not depend on Deno, and this branch leaves it unchanged.

## The nodejs-wheel package

`nodejs-wheel` puts official Node.js builds on PyPI. The `nodejs-wheel-binaries` package holds only the binary and is the one DSPy should depend on. The `nodejs-wheel` package adds console scripts on top.

- The latest stable release is 24.19.0, which tracks Node.js 24 LTS. Node.js 25 and 26 are published only as release candidates, e.g., `26.7.0rc0`.
- It has wheels for macOS x86-64 and arm64, glibc and musl Linux x86-64 and arm64, and Windows x86-64 and arm64. That covers more platforms than the `deno` wheel, which has no musl or Windows arm64 builds.
- The download is about 56 to 61 MB. The `deno` wheel is about 44 to 50 MB.
- The installed size is about 200 MB because it includes npm, corepack, and C headers. Deno installs as a single binary.
- One person maintains it under the MIT license. It is not an official Node.js project.

The binary lives at `nodejs_wheel/bin/node`, or at `nodejs_wheel/node.exe` on Windows. The package has no `find_node_bin()` helper, so DSPy computes that path itself.

## What changed on this branch

### Runtime discovery and version check

DSPy looks for the binary from `nodejs_wheel` first and then for `node` on `PATH`. It requires Node.js 24 or newer. Node.js 22 fails because its V8 lacks the current WebAssembly JSPI interface, which Pyodide's `run_sync` needs for tool calls. Node.js 24 has JSPI behind the `--experimental-wasm-jspi` flag. Node.js 25 and newer turn it on by default and reject the flag, so DSPy adds the flag only for Node.js 24.

### Getting Pyodide

Node.js cannot import from npm by name the way Deno can. DSPy now downloads the pinned Pyodide tarball from the npm registry on first use. It checks the tarball against the registry's SHA-512 checksum and unpacks it to `~/.dspy_cache/pyodide/0.29.4/`. The `DSPY_PYODIDE_DIR` variable points DSPy at a local copy for offline use. The download is 12.6 MB unpacked. Deno also downloads Pyodide on first run, so this is not a new network dependency.

The runner loads Pyodide by file path. A `package.json` or `node_modules` directory near the working directory therefore cannot change which Pyodide loads. This replaces the `--no-config`, `--no-lock`, and `DENO_NO_PACKAGE_JSON` handling.

### Process flags

DSPy starts Node.js with these flags:

- `--permission` turns on the permission model. Without grants, the process cannot read or write files, spawn child processes, start workers, or load native addons.
- `--allow-fs-read` grants the runner, the Pyodide directory, and the user's read and write paths.
- `--allow-fs-write` grants the user's write paths.
- `--disallow-code-generation-from-strings` turns off `eval` and `new Function` in JavaScript.
- `--allow-net` is added on Node.js 25 and newer when the user sets `enable_network_access`.

### Environment variables

Node.js has no permission for environment variables. DSPy now starts the process with only the variables listed in `enable_env_vars`. This also drops `NODE_OPTIONS`, which could otherwise load extra code into the runtime.

### The restricted `js` module

Pyodide's `js` module gives Python a handle on the JavaScript global object. Under Deno, the sandbox could reach `js.Deno`, and Deno's permissions stopped it from doing harm. Under Node.js, the same handle would give Python `process`, `process.getBuiltinModule`, and `fetch`.

The runner now passes Pyodide a small `jsglobals` object. It holds only the timer functions. It also holds `fetch` and the related classes when the user enables network access. Python code that asks for `js.process` or `js.fetch` gets an `AttributeError`. With code generation turned off, `pyodide.code.run_js` and the `Function` constructor fail too.

### Runner changes

`runner.js` became `runner.mjs`. The protocol code did not change. The runtime calls changed:

- `Deno.stdin` with `readLines` became `node:readline`.
- `Deno.readFile` and `Deno.writeFile` became `fs.promises`.
- `Deno.env.get` became `process.env`.
- The `unhandledrejection` listener became `process.on("unhandledRejection")`.
- The `Deno.permissions.revoke` call is gone. The Pyodide cache directory holds only Pyodide, so nothing in it needs protecting.

The runner also replaces `process.binding("constants")` with a small function. Pyodide's Emscripten file system code calls it during start up to read the file open flags. The permission model blocks `process.binding`, and the replacement returns the same public `fs.constants` value.

The runner adds an `uncaughtException` handler. Node.js prints the source line of a fatal error, and Pyodide's source is one minified line longer than the pipe buffer. Without the handler, the error message is cut off before it reaches Python.

### Public API

- `node_command` replaces `deno_command`. Passing `deno_command` raises a `TypeError` that names the new argument.
- `node_process` replaces the `deno_process` attribute.
- The `dspy[deno]` extra is now `dspy[node]`.

## What we lose

### Deno's security guarantee

Deno treats a permission bypass as a security bug. The Node.js permission documentation says the model is a "seat belt" for trusted code and "does not provide security guarantees in the presence of malicious code". Model-generated code is untrusted, so the Node.js layer alone is weaker than what DSPy has today.

The restricted `js` module is the main defense on this branch. It stops sandboxed code from reaching Node.js objects at all. The permission model is now a second layer behind it. The second layer still catches some escapes. Pyodide exposes `pyodide_js.FS.mount` with its `NODEFS` file system, which reads the host disk through Node.js. A test on this branch confirms that Node.js denies that mount without a grant.

### Network limits on Node.js 24

Node.js 24 has no network permission. On Node.js 24, the only network control is that sandboxed code has no `fetch`. Python's `socket` module in Pyodide cannot open connections, and `urllib` crashes the runtime. Node.js 25 added `--allow-net`, and DSPy uses it there.

### Per-host network allow lists

Deno accepts `--allow-net=api.example.com`. Node.js `--allow-net` is all or nothing, and the flag is still marked experimental. When a user passes `enable_network_access=["api.example.com"]`, DSPy now grants access to every host and logs a warning. Users who need a per-host list would have to use a proxy or a different interpreter.

### Signals to other processes

Deno requires `--allow-run` for `Deno.kill`. Node.js does not restrict `process.kill`. The restricted `js` module hides `process`, so sandboxed code cannot reach it on this branch. Without that module restriction, it could.

### Clean error messages for some escape attempts

Some blocked attempts crash the runtime instead of raising a Python exception, e.g., calling the `Function` constructor. DSPy reports these as a `CodeInterpreterError` and ends the session. That keeps the sandbox closed, but the message reads `PythonError` with no detail.

### Install size

The installed `nodejs-wheel-binaries` is about four times larger on disk than the `deno` wheel.

## What we gain

- Node.js is the most widely installed JavaScript runtime. Many users already have Node.js 24 on their machines.
- Pyodide's main test target is Node.js. Pyodide bugs on Node.js are more likely to be found and fixed upstream.
- The interpreter starts in about the same time. On an M-series Mac, the median cold start was 0.86 seconds with Node.js 24 and 0.99 seconds with Deno 2.6.
- The two interpreter test files ran in 20 seconds with Node.js and 27 seconds with Deno.
- The wheel covers musl Linux and Windows arm64.

## Test results

On this branch with Node.js 24.19.0:

- `tests/primitives`, `tests/predict`, `tests/flex`, and `tests/callback` with `--deno` gave 950 passed and 91 skipped.
- The default suite gave 2219 passed. Thirteen LM proxy server and streaming tests fail, and they fail the same way on `main`.

The code execution suites also pass on Node.js 26.7.0rc0. Node.js 22.23 fails the version check with a clear message.

Tests that checked Deno details now check the Node.js equivalents. New tests cover these cases:

- The sandbox cannot reach `js.process` or generate JavaScript.
- A `NODEFS` mount without a grant fails.
- Unlisted environment variables and `NODE_OPTIONS` do not reach the process.
- A tampered Pyodide download fails its checksum.
- The JSPI and network flags follow the Node.js version.

## Open questions

1. Is the restricted `js` module plus the Node.js permission model a strong enough sandbox for untrusted code? A focused review of the Pyodide objects that remain reachable, e.g., `pyodide_js._api` and `pyodide_js._module`, would answer this.
2. Should DSPy require Node.js 25 or newer so network limits apply at the process level? That would mean depending on `nodejs-wheel-binaries` release candidates until a stable 26.x wheel ships.
3. Should DSPy run the sandbox inside an operating system sandbox as well, e.g., `bwrap` on Linux or `sandbox-exec` on macOS? The Node.js documentation recommends this for untrusted code.
4. Should DSPy keep a Deno path for one release so users can move over at their own pace?
5. Should DSPy ship Pyodide in a separate wheel so first use needs no npm registry access?

## Follow-up work before merging

- Regenerate `uv.lock`. Running `uv lock` with a newer uv rewrites most of the file, so this should happen with the uv version the project uses.
- Rename the `deno` pytest marker and the `--deno` flag. About 15 test files, `tests/conftest.py`, and the CI workflows use them.
- Update the remaining Deno references in `docs/docs/api/modules/RLM.md`, `docs/docs/api/modules/Flex.md`, `docs/docs/diving-deeper/rlm.md`, `docs/docs/diving-deeper/flex.md`, and `docs/docs/diving-deeper/built-in-module-variants.md`.
- Add a release note that names the `deno_command` removal and the `dspy[node]` extra.
- Test on Windows. The binary path logic follows the `nodejs_wheel` layout, but nobody has run it there.
