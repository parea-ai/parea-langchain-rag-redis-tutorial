from parea.evals.general import answer_matches_target_llm_grader_factory
from parea.evals.rag import (
    context_query_relevancy_factory,
    percent_target_supported_by_context_factory,
)
from parea.evals.utils import EvalFuncTuple, run_evals_in_thread_and_log
from parea.schemas.log import Log, LLMInputs


# a selection of pre-defined evaluation functions from Parea AI. A EvalFuncTuple contains the name of the evaluation
# function and an eval function (or a prebuild factory that returns an eval function).
EVALS = [
    EvalFuncTuple(
        name="matches_target",
        func=answer_matches_target_llm_grader_factory(),
    ),
    # For our prompt template we use the keyword "context" to refer to retrieved context
    EvalFuncTuple(
        name="relevancy",
        func=context_query_relevancy_factory(context_fields=["context"]),
    ),
    EvalFuncTuple(
        name="supported_by_context",
        func=percent_target_supported_by_context_factory(context_fields=["context"]),
    ),
]


def run_evals(
    trace_id: str,
    question: str,
    context: str,
    response: str,
    target_answer: str,
    model_name: str = "gpt-3.5-turbo-16k",
    provider: str = "openai",
    verbose: bool = True,
):
    """
    Run evaluation metrics in a background thread on the provided response to a question and context.
    :param trace_id: trace_id of the LLM call
    :param question: question asked
    :param context: context used to answer the question
    :param response: response to the question
    :param target_answer: target answer to the question
    :param model_name: model name
    :param provider: provider
    :param verbose: whether to print evaluation results

    :return: None
    """
    # build log component needed for evaluation metric functions
    log = Log(
        configuration=LLMInputs(model=model_name, provider=provider),
        inputs={"question": question, "context": context},
        output=response,
        target=target_answer,
    )

    # helper function to run evaluation metrics in a thread to avoid blocking return of chain
    run_evals_in_thread_and_log(
        trace_id=trace_id, log=log, eval_funcs=EVALS, verbose=verbose
    )
