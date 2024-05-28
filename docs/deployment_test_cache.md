# Deployment Test Cache

The deployment test cache is a system to enable ML models to be 

There are a few environment variables that can be set to control the behavior of the tests:
- `USE_DEPLOYMENT_CACHE`: If set to `true`, the tests will use the deployment cache to avoid downloading the models and running the deployments. This is useful for running integration tests faster and in the environment where GPU is not available.
- `SAVE_DEPLOYMENT_CACHE`: If set to `true`, the tests will save the deployment cache after running the deployments. This is useful for updating the deployment cache if new deployments or tests are added.

### How to use the deployment cache environment variables

Here are some examples of how to use the deployment cache environment variables.

#### Do you want to run the tests normally using GPU?
    
```bash
USE_DEPLOYMENT_CACHE=false
SAVE_DEPLOYMENT_CACHE=false
```

This is the default behavior. The tests will run normally using GPU and the deployment cache will be completely ignored.

#### Do you want to run the tests faster without GPU?

```bash
USE_DEPLOYMENT_CACHE=true
SAVE_DEPLOYMENT_CACHE=false
```

This will run the tests using the deployment cache to avoid downloading the models and running the deployments. The deployment cache will not be updated after running the deployments. Only use it if you are sure that the deployment cache is up to date.

#### Do you want to update the deployment cache?

```bash
USE_DEPLOYMENT_CACHE=false
SAVE_DEPLOYMENT_CACHE=true
```

This will run the tests normally using GPU and save the deployment cache after running the deployments. Use it if you have added new deployments or tests and want to update the deployment cache.
