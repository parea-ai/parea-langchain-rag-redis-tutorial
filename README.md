# rag-redis-parea

This template performs RAG using Redis (vector database) and OpenAI (LLM) on financial 10k filings docs for Nike.

It relies on the sentence transformer `all-MiniLM-L6-v2` for embedding chunks of the pdf and user questions.

It uses Parea AI to instrument tracing and evaluations.

*Updated from the original LangChain
template [rag-redis](https://github.com/langchain-ai/langchain/tree/master/templates/rag-redis).*

## Environment Setup

_Copy the `.env.example` file to `.env` and fill in the values. Or export values in shell._

Set the `OPENAI_API_KEY` environment variable to access the [OpenAI](https://platform.openai.com) models.

Set the `PAREA_API_KEY` environment variable to access tracing
with [PareaAI](https://docs.parea.ai/integrations/langchain):

```bash
export OPENAI_API_KEY= <YOUR OPENAI API KEY>
export PAREA_API_KEY= <YOUR OPENAI API KEY>
```

Set the following [Redis](https://redis.com/try-free) environment variables:

```bash
export REDIS_HOST = <YOUR REDIS HOST> 
export REDIS_PORT = <YOUR REDIS PORT>
export REDIS_PASSWORD = <YOUR REDIS PASSWORD>
```

## Supported Settings

We use a variety of environment variables to configure this application

| Environment Variable     | Description                   | Default Value |
|--------------------------|-------------------------------|---------------|
| `REDIS_HOST`             | Hostname for the Redis server | "localhost"   |
| `REDIS_PORT`             | Port for the Redis server     | 6379          |
| `REDIS_PASSWORD`         | Password for the Redis server | ""            |
| `INDEX_NAME`             | Name of the vector index      | "rag-redis"   |
| `TOKENIZERS_PARALLELISM` | To avoid potential deadlocks  | "False"       |

## Usage

Load requirements:

```shell
poetry install
```

Start the [Redis server](https://redis.io/docs/install/install-stack/):

```shell
redis-stack-server
```

Then, from the root directory run ingest-docs using the CLI helper to load your data into Redis.

```shell
python main.py --ingest-docs
```

Then run the chain (use --run-eval to also run evaluations defined in evals/evals.py):

```shell
python main.py --run-eval
```

## Results

View trace logs on [Parea AI](https://app.parea.ai/logs).