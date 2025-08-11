# Code Standards

This project uses Ruff for linting and formatting. If you want to 
manually run Ruff on the codebase, using uv it's

```sh
uv run ruff check aana
```

You can automatically fix some issues with the `--fix`
 and `--unsafe-fixes` options. (Be sure to install the dev 
 dependencies: `uv sync --group dev`. )

To run the auto-formatter, it's

```sh
uv run ruff format aana
```

(If you are running code in a non-uv environment, just leave off `uv run`.)

For users of VS Code, the included `settings.json` should ensure
that Ruff problems appear while you edit, and formatting is applied
automatically on save.
