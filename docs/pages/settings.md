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


# Settings

Here are the environment variables that can be used to configure the Aaana SDK:

- TMP_DATA_DIR: The directory to store temporary data. Default: `/tmp/aana`.
- NUM_WORKERS: The number of request workers. Default: `2`.
- DB_CONFIG: The database configuration in the format `{"datastore_type": "sqlite", "datastore_config": {"path": "/path/to/sqlite.db"}}`. Currently only SQLite and PostgreSQL are supported. Default: `{"datastore_type": "sqlite", "datastore_config": {"path": "/var/lib/aana_data"}}`.
- USE_DEPLOYMENT_CACHE (testing only): If set to `true`, the tests will use the deployment cache to avoid downloading the models and running the deployments. Default: `false`.
- SAVE_DEPLOYMENT_CACHE (testing only): If set to `true`, the tests will save the deployment cache after running the deployments. Default: `false`.
- HF_HUB_ENABLE_HF_TRANSFER: If set to `1`, the HuggingFace Transformers will use the HF Transfer library to download the models from HuggingFace Hub to speed up the process. Recommended to always set to it `1`. Default: `0`.
- HF_TOKEN: The HuggingFace API token to download the models from HuggingFace Hub, required for private or gated models.
