# Testing

The project uses pytest for testing. To run the tests, use the following command:

```bash
poetry run pytest
```

If you are using VS Code, you can run the tests using the Test Explorer that is installed with the [Python extension](https://code.visualstudio.com/docs/python/testing).

Testing ML models poses a couple of problems: loading and running models may be very time consuming, and you may wish to run tests on systems that lack hardware support necessary for the models, for example a subnotebook without a GPU or a CI/CD server. To solve this issue, we created a **deployment test cache**. See [deployment test cache docs](deployment_test_cache.md) for more information.
