import argparse

from ingest import ingest_documents
from rag.chain import run_chain

parser = argparse.ArgumentParser(description="CLI Helper")
parser.add_argument(
    "-q",
    "--question",
    type=str,
    help="The question",
    default="Which operating segment contributed least to total Nike brand revenue in fiscal 2023?",
)
parser.add_argument(
    "-t",
    "--target",
    type=str,
    help="The target answer",
    default="Global Brand Divisions",
)
parser.add_argument(
    "--run-eval",
    action="store_true",
    help="Run evals if specified",
)
parser.add_argument(
    "--ingest-docs",
    action="store_true",
    help="Ingest documents if specified. You need to run this before running the chain.",
)
args = parser.parse_args()

if args.ingest_docs:
    ingest_documents()
else:
    run_chain(args.question, args.target, args.run_eval)
