# Aana SDK Getting Started

This document is a draft. The final version will live on Github.


## Code overview

aana/ - top level source code directory for the project

    alembic/ - directory for database automigration

         versions/ - individual migrations

    api/ - code for generating an API from deployment and endpoint configurations

    config/ - various configuration, including settings, but also config for deployments & endpoints

         build.py - config for building the SDK pipeline

         db.py - config for the database

         deployments.py - config for available deployments

         endpoints.py - deployment names and associated endpoints

         pipeline.py - pipeline nodes for SDK

         settings.py - app settings

    deployments/ - classes for individual model deployments, wrap the model's functionality

    exceptions/ - custom exception classes

    models/ - model classes for data

         core/ - core data models for the SDK

         db/ - database models 

         pydantic/ - pydantic models for HTTP inputs and outputs

    repository/ - repository classes for storage

         datastore/ - repositories for structured databases (SQL)

         vectorstore/ - repositories for vector databases (Qdrant) (TODO)

    tests/ - automated tests for the SDK

         db/ - tests for database functions

         deployments/ - tests for model deployments

         files/ - files used for tests

     utils/ - various utility functionality

           chat_templates/ - templates for LLM chats

     main.py - entry point for the application


## Adding a New Model

A deployment is a standardized interface for any kind of functionality that needs to manage state (primarily, but not limited to, an AI model that needs to be fetched and loaded onto a GPU). New deployments should inherit from aana.deployments.base_deployment.BaseDeployment, and be in the aana/deployments folder. Additionally, a deployment config for the deployment will have to be added to the aana/configs/deployments.py file. Some examples:

aana/configs/deployments.py:

aana/deployments/foo_0b_deployment.py:

Now you have a deployment! Unfortunately, that's not enough to run it just yet.


## Adding Pipeline Nodes

Next you need to add some pipeline nodes to aana/config/pipeline.py so you can use your new deployment. A typical minimal, 1-stage inference workflow might be:

1. input - says which input to get from a request
2. preprocessing - turning the HTTP request into something you can use (download files, etc)
3. inference - running on the ML model
4. postprocessing - making the response something useful for a human to see (sometimes included in inference)

(Not shown for simplicity: video processing params on the input, saving to the database)

![](diagram.png)

Here's an example of these for a video processing pipeline(aana/config/pipeline.py):


## Adding endpoints

Now we're almost done. The last stage is to add a run target with endpoints that refer to the node inputs and outputs (aana/config/endpoints.py).

Okay, can you figure out where there might be a bug?

We created a postprocessing step to save the video captions to the DB, but since we didn't include its output in the endpoint definition ("captions_ids"), that step **won't** run. It will get the output of the captions model, determine that no more steps are needed, and return it to the user. A working endpoint that wanted to include `EndpointOutput(name="caption_ids", output="caption_ids")` in the list of outputs.


## Saving to a DB

Aana SDK is designed to have two databases, a structured database layer with results/metadata (datastore) and a vector database for search, etc (vectorstore).


### Saving to datastore

You will need to add database entity models to a class file aana/models/db/. Additionally, to avoid import issues, you will need to import that model inside aana/models/db/__init__.py.

Once you have defined your model, you will need to create an alembic migration to create the necessary table and modify other tables if necessary. Do this just by running

The app will automatically run the migration when you start up, so the rest is taken care of.


### Repositories and Helper functions

We wrap access to the datastore in a repository class. There is a generic BaseRepository that provides the following methods: create, create_multiple, read (by id), delete (by id). If you want to fetch by another parameter (for example by a parent object id). Update logic is TODO since the semantics of in-place updates with the SqlAlchemy ORM is a bit complex.

(aana/repository/datastore/caption_repository.py)

That's it! 

To make a repository work with the pipeline, it's easiest to wrap the repository actions in helper functions, like so (aana/utils/db.py):


## Vectorstore

TODO


### Running the SDK

So, you have created a new deployment for your model, defined the pipeline nodes for it, and created the endpoints to call it. How do you run the SDK?

The "normal" way to run the SDK right now is to call the SDK as a module with following syntax: \
 \


Host and port are optional; the defaults are `0.0.0.0` and `8000`, respectively, but you must give a deployment name as the target or the SDK doesn't know what to run. One the SDK has initialized the pipeline, downloaded any remote resources necessary, and loaded the model weights, it will print "Deployed Serve app successfully." and that is the cue that it is ready to serve traffic and perform tasks, including inference.


## Tests

Write unit tests for every freestand function or task function you add. Write unit tests for deployment methods or static functions that transform data without sending it to the inference engine (and refactor the deployment code so that as much functionality as possible is modularized so that it may be tested). 

Additionally, please write some tests for the `tests/deployments` folder that will load your deployment and programmatically run some inputs through it to validate that the deployment itself works as expected. Note, however, that due to the size and complexity of loading deployments that this might fail even if the logic is correct, if for example the user is running on a machine without a GPU.

Additionally, you can use mocks to test things like database logic, or other code that would normally require extensive external functionality to test. For example, here is code that mocks out database calls so it can be run as a deployment without needed to load the model:
