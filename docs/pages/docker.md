# Run with Docker

We provide a docker-compose configuration to run the application in a Docker container in [Aana App Template](https://github.com/mobiusml/aana_app_template/?tab=readme-ov-file#running-with-docker).

Requirements:

- Docker Engine >= 26.1.0
- Docker Compose >= 1.29.2
- NVIDIA Driver >= 525.60.13

You can edit the [Dockerfile](https://github.com/mobiusml/aana_app_template/Dockerfile.yaml) to assemble the image as you desire and
and [docker-compose file](https://github.com/mobiusml/aana_app_template/docker-compose.yaml) for container instances and their environment variables.

To run the application, simply run the following command:

```bash
docker-compose up
```

The application will be accessible at `http://localhost:8000` on the host server.


!!! warning

    If your applications requires GPU to run, you need to specify which GPU to use.

    The applications will detect the available GPU automatically but you need to make sure that `CUDA_VISIBLE_DEVICES` is set correctly.
    
    Sometimes `CUDA_VISIBLE_DEVICES` is set to an empty string and the application will not be able to detect the GPU. Use `unset CUDA_VISIBLE_DEVICES` to unset the variable.

    You can also set the `CUDA_VISIBLE_DEVICES` environment variable to the GPU index you want to use: `CUDA_VISIBLE_DEVICES=0 docker-compose up`.


!!! Tip

    Some models use Flash Attention for better performance. You can set the build argument `INSTALL_FLASH_ATTENTION` to `true` to install Flash Attention. 

    ```bash
    INSTALL_FLASH_ATTENTION=true docker-compose build
    ```

    After building the image, you can use `docker-compose up` command to run the application.

    You can also set the `INSTALL_FLASH_ATTENTION` environment variable to `true` in the `docker-compose.yaml` file.

