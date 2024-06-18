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


# Testing

The project uses pytest for testing. To run the tests, use the following command:

```bash
poetry run pytest
```

If you are using VS Code, you can run the tests using the Test Explorer that is installed with the [Python extension](https://code.visualstudio.com/docs/python/testing).

Testing ML models poses a couple of problems: loading and running models may be very time consuming, and you may wish to run tests on systems that lack hardware support necessary for the models, for example a subnotebook without a GPU or a CI/CD server. To solve this issue, we created a **deployment test cache**. See [the documentation](docs/deployment_test_cache.md).