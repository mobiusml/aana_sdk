---
hide:
  - navigation
---

<style>
.md-content .md-typeset h1 { 
  position: absolute;
  left: -999px;
}
</style>


# Code Standards

This project uses Ruff for linting and formatting. If you want to 
manually run Ruff on the codebase, using poetry it's

```sh
poetry run ruff check aana
```

You can automatically fix some issues with the `--fix`
 and `--unsafe-fixes` options. (Be sure to install the dev 
 dependencies: `poetry install --with=dev`. )

To run the auto-formatter, it's

```sh
poetry run ruff format aana
```

(If you are running code in a non-poetry environment, just leave off `poetry run`.)

For users of VS Code, the included `settings.json` should ensure
that Ruff problems appear while you edit, and formatting is applied
automatically on save.
