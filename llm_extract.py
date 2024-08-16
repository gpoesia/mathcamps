import argparse
import json
from evaluate import \
    infer_answer_extractor, extract_answer, grade_reasoning,\
    eval_results, grade_prediction, parse_predictions_and_answers
from grammar import load_standards
import copy
import os
import re

import numpy as np
import torch
from vllm import LLM, SamplingParams


with open('prompts/regrade.json', 'r') as f:
    REGRADE_PROMPTS = json.load(f)


# input: ground truth answer
# output: shots for GPT to use to grade answers of that type
def _infer_answer_shots(answer):
    if answer in "<=>":
        return REGRADE_PROMPTS["compare"]
    if answer.startswith('{') or answer.startswith('['):
        # Grading a set of predictions, like in a system of equations.
        return REGRADE_PROMPTS["list"]
    if '/' in answer:
        return REGRADE_PROMPTS["fraction"]

    return REGRADE_PROMPTS["default"]


def extract_llm_answer_core(answer):
    # Search for 
    return None if "none" in answer.lower() else answer


def save_results(output_path, results):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def extract_answer_core(answer: str) -> str:
    return re.search('([0-9.\\-\\+>=<][0-9.\\-\\+/ ]*%?)', answer).group(0)


def grade(answer, llm_answer):
    try:
        if grade_prediction(answer, llm_answer):
            return True
    except Exception:
        pass

    try:
        if grade_prediction(answer, extract_answer_core(llm_answer)):
            return True
    except Exception:
        pass

    try:
        return eval(answer) == eval(llm_answer)
    except Exception:
        return False


def _encode_header(message: list[dict]) -> str:
    return "<|start_header_id|>" + message["role"] + "<|end_header_id|>\n\n"

def _encode_message(message: dict) -> str:
    return _encode_header(message) + message["content"].strip() + "<|eot_id|>"

def format_prompt_for_chat_lm(dialog: list[dict]) -> str:
    # Reference: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L203
    return ("<|begin_of_text|>" +
            "".join([_encode_message(message) for message in dialog]) +
            _encode_header({"role": "assistant"}))


def llm_regrade(results, llm=None, regrader_llm='meta-llama/Meta-Llama-3-70B') -> int:
    regrading_prompts = {}

    for k, v in results.items():
        # Run through lm if main answer incorrect or None
        if not v['correct']:
            prompt_messages = _infer_answer_shots(v['answer'])
            curr_reasoning = [{"role": "user", "content": v['model_generation']}]
            regrading_prompts[k] = prompt_messages + curr_reasoning

        for k_fup, v_fup in v['followups'].items():
            if not v_fup['correct']:
                prompt_messages = _infer_answer_shots(v_fup['answer'])
                curr_reasoning = [{"role": "user", "content": v_fup['model_generation']}]
                regrading_prompts[k_fup] = prompt_messages + curr_reasoning

    if not llm:
        llm = LLM(model=regrader_llm,
                  tensor_parallel_size=torch.cuda.device_count())

    keys_prompts = [(k, format_prompt_for_chat_lm(v))
                    for k, v in regrading_prompts.items()]
    keys, prompts = [k for k, _ in keys_prompts], [p for _, p in keys_prompts]

    responses = llm.generate(prompts, SamplingParams(
        temperature=0.0,
        stop=["<|eot_id|>", "|||"],
        max_tokens=30,
    ))

    outputs = [r.outputs[0].text for r in responses]
    # outputs = v['llm_extracted_answer']
    outputs_by_key = dict(zip(keys, outputs))

    n_changes = 0

    for k, v in results.items():
        if not v['correct']:
            llm_answer = extract_llm_answer_core(outputs_by_key[k])
            v['llm_extracted_answer'] = llm_answer
            correct = llm_answer and grade(v['answer'], llm_answer)

            if correct:
                v['correct'] = True
                n_changes += 1

        for k_fup, v_fup in v['followups'].items():
            if not v_fup['correct']:
                llm_fup_answer = extract_llm_answer_core(outputs_by_key[k_fup])
                v_fup['llm_extracted_answer'] = llm_fup_answer
                correct = llm_fup_answer and grade(v_fup['answer'], llm_fup_answer)

                if correct:
                    v_fup['correct'] = True
                    n_changes += 1

    return n_changes


def regrade_results_from_paths(results_paths):
    for path in results_paths:
        with open(path, 'r') as f:
            results = json.load(f)
            n_changes = llm_regrade(results)
            print(f"Made {n_changes} changes to {path}")
            # Save 'a/b/c.json -> 'a/b/regraded-c.json'
            new_path = os.path.join(os.path.dirname(path), 'regraded-' + os.path.basename(path))
            save_results(new_path, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reextract', action='store_true', help='Reextract final answers from LLM reasoning using Llama.')
    parser.add_argument('--results', nargs='*', help='JSON file(s) containing results to be regraded.')

    opt = parser.parse_args()

    if opt.reextract:
        regrade_results_from_paths(opt.results)


if __name__ == '__main__':
    main()
