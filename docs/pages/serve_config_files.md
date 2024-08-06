# Serve Config Files

The [Serve Config Files](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-config-files) is the recommended way to deploy and update your applications in production. Aana SDK provides a way to build the Serve Config Files for the Aana applications.

## Building Serve Config Files

To build the Serve config file, run the following command:

```bash
aana build <app_module>:<app_name>
```

For example:

```bash
aana build aana_chat_with_video.app:aana_app
```

The command will generate the Serve Config file and App Config file and save them in the project directory. You can then use these files to deploy the application using the Ray Serve CLI.

## Deploying with Serve Config Files

When you are running the Aana application using the Serve config files, you need to run the migrations to create the database tables for the application. To run the migrations, use the following command:

```bash
aana migrate <app_module>:<app_name>
```

For example:

```bash
aana migrate aana_chat_with_video.app:aana_app
```

Before deploying the application, make sure you have the Ray cluster running. If you want to start a new Ray cluster on a single machine, you can use the following command:

```bash
ray start --head
```

For more info on how to start a Ray cluster, see the [Ray documentation](https://docs.ray.io/en/latest/ray-core/starting-ray.html#starting-ray-via-the-cli-ray-start).

To deploy the application using the Serve config files, use [`serve deploy`](https://docs.ray.io/en/latest/serve/advanced-guides/deploy-vm.html#serve-in-production-deploying) command provided by Ray Serve. For example:

```bash
serve deploy config.yaml
```
