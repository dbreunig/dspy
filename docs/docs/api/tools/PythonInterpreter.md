# dspy.PythonInterpreter

## Node.js Installation

`PythonInterpreter` uses Node.js and Pyodide to run Python in a local WASM sandbox. The recommended installation
keeps Node.js in the same Python environment as DSPy:

```bash
pip install "dspy[node]"
```

The `node` extra installs `nodejs-wheel-binaries` (`>=24.0.0`), a community-maintained wheel of the official
Node.js release. DSPy prefers that managed binary when it is installed, so Python dependency locking also locks the
Node.js runtime. The extra provides binaries for macOS x86-64/arm64, glibc and musl Linux x86-64/arm64, and
Windows x86-64/arm64. It downloads approximately 55–60 MiB.

On other platforms, install Node.js 24 or newer from [nodejs.org](https://nodejs.org/en/download). DSPy falls back
to the `node` executable on `PATH`. An explicit `node_command` passed to `PythonInterpreter` takes precedence over
both options.

The first interpreter downloads the pinned Pyodide release from the npm registry, verifies its checksum, and
stores it under `~/.dspy_cache/pyodide/`. Set `DSPY_PYODIDE_DIR` to an unpacked copy of the `pyodide` npm package
to run without network access. The runner loads Pyodide from that directory by path, so a `package.json` or
`node_modules` directory near the current working directory cannot redirect it.

### Sandbox Boundary

DSPy starts Node.js with its permission model enabled. The process can read only the runner, the Pyodide files, and
the paths in `enable_read_paths` and `enable_write_paths`. It can write only to `enable_write_paths`, and it cannot
spawn processes, start workers, or load native addons. The process receives only the environment variables named
in `enable_env_vars`.

Sandboxed Python sees a restricted `js` module. It has no access to Node's `process`, module loader, or built-in
modules, and JavaScript code generation from strings is disabled. Network globals such as `fetch` are present only
when `enable_network_access` is set. Node.js 25 and newer also enforce network access at the process level. Node.js
cannot limit network access to particular hosts, so DSPy grants access to every host and logs a warning when
`enable_network_access` is set.

## Execution Instructions

`PythonInterpreter.execution_instructions` describes stable constraints of its Pyodide execution environment. It
is class metadata, so code-generating modules can inspect it without starting Node.js or allocating an interpreter.
`RLM` includes these instructions in its action prompt, which adapters render in the model's system prompt.

Custom interpreter factories may expose their own `execution_instructions` string. This metadata is optional; a
factory without it remains valid and uses RLM's generic action prompt.

<!-- START_API_REF -->
::: dspy.PythonInterpreter
    handler: python
    options:
        members:
            - __call__
            - execute
            - shutdown
            - start
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
