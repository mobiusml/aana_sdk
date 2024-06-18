# Run with Docker

You can deploy example applications using Docker. 

1. Clone the repository.

2. Build the Docker image.

```bash
docker build -t aana:latest .
```

3. Run the Docker container.

```bash
docker run --rm --init -p 8000:8000 --gpus all -e TARGET="llama2" -v aana_cache:/root/.aana -v aana_hf_cache:/root/.cache/huggingface --name aana_instance aana:latest
```

Use the environment variable TARGET to specify the application you want to run. The available applications are `chat_with_video`, `whisper`, and `llama2`.

The first run might take a while because the models will be downloaded from the Internet and cached. The models will be stored in the `aana_cache` volume. The HuggingFace models will be stored in the `aana_hf_cache` volume. If you want to remove the cached models, remove the volume.

Once you see `Deployed successfully.` in the logs, the server is ready to accept requests.

You can change the port and gpus parameters to your needs.

The server will be available at http://localhost:8000.

The app documentation available as a [Swagger UI](http://localhost:8000/docs) and [ReDoc](http://localhost:8000/redoc).

5. Send a request to the server.

You can find examples in the [demo notebook](notebooks/demo.ipynb).