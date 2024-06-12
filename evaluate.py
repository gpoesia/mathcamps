#!/usr/bin/env python3

import argparse
import time
import json
import yaml
import re
import os
import gc
import collections
import signal
from typing import Optional
from contextlib import contextmanager
from fractions import Fraction as F
import requests
import traceback

from tqdm import tqdm

import torch
from openai import OpenAI
import openai
from together import Together
import together
from anthropic import Anthropic
import anthropic
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason, Content
import vertexai.preview.generative_models as generative_models
import google
from vllm import LLM, SamplingParams



try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except:
    print('Could not import huggingface - will not be able to run inference on those models')
#import torch


api_calls_log = open('api_calls.log.json', 'a')

def log_api_call(ctype: str, m: str, s):
    api_calls_log.write(json.dumps({'type': ctype, 'model': m, 'content': json.dumps(s)}))
    api_calls_log.write('\n')
    api_calls_log.flush()

# Adapted from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)

MESSAGE_DELIMETER = '@@@'


def format_prompt_for_completion_lm(prompt_messages: list[dict]) -> str:
    lines = []

    for m in prompt_messages:
        lines.append(f'|{m["role"]}| {m["content"]}')
        lines.append(MESSAGE_DELIMETER)

    prompt = '\n'.join(lines) + "|assistant|"
    return prompt


class LanguageModel:
    def name(self):
        raise NotImplementedError

    def predict(self, prompt_messages: list[dict], temperature=0, max_tokens=1000) -> str:
        raise NotImplementedError

    def is_local(self) -> bool:
        return False


class OpenAIChatModel:
    def __init__(self, name: str, model_str: str):
        self._name = name
        self._model_str = model_str

    def name(self):
        return self._name

    def model_str(self):
        return self._model_str

    def predict(self, prompt_messages: list[dict], temperature=0, max_tokens=1000) -> str:
        client = OpenAI()
        while True:
            try:
                with time_limit(10):
                    log_api_call('request', self._model_str, prompt_messages)
                    response = client.chat.completions.create(model=self._model_str,
                                                            messages=prompt_messages,
                                                            temperature=temperature,
                                                            max_tokens=max_tokens)
                    log_api_call('response', self._model_str, response.choices[0].message.content)
                    return response.choices[0].message.content
            except TimeoutError:
                print('OpenAI API timed out, retrying...')
            except openai.OpenAIError as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)


class TogetherAIChatModel:
    def __init__(self, name: str, model_str: str):
        self._name = name
        self._model_str = model_str

    def name(self):
        return self._name

    def model_str(self):
        return self._model_str

    def predict(self, prompt_messages: list[dict], temperature=0, max_tokens=1000) -> str:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        while True:
            try:
                with time_limit(30):
                    response = client.chat.completions.create(
                        model=self._model_str,
                        messages=prompt_messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content
            except TimeoutError:
                print('TogetherAI API timed out, retrying...')
            except openai.OpenAIError as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)
            except TimeoutError as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)
            except requests.exceptions.ReadTimeout as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)
            except Exception as e:
                traceback.print_exc()
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)


class TogetherAIModel:
    def __init__(self, name: str, model_str: str):
        self._name = name
        self._model_str = model_str

    def name(self):
        return self._name

    def model_str(self):
        return self._model_str

    def predict(self, prompt_messages: list[dict], temperature=0, max_tokens=1000) -> str:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        while True:
            try:
                prompt = format_prompt_for_completion_lm(prompt_messages)
                with time_limit(30):
                    response = client.completions.create(
                        model=self._model_str,
                        prompt=prompt
                    )
                    return response.choices[0].text
            except TimeoutError:
                print('TogetherAI API timed out, retrying...')
            except openai.OpenAIError as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)
            except TimeoutError as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)
            except requests.exceptions.ReadTimeout as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)
            except Exception as e:
                traceback.print_exc()
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)

class AnthropicModel:
    def __init__(self, name: str, model_str: str):
        self._name = name
        self._model_str = model_str

    def name(self):
        return self._name

    def model_str(self):
        return self._model_str

    def predict(self, prompt_messages: list[dict], temperature=0, max_tokens=1000) -> str:
        client = Anthropic()
        while True:
            try:
                with time_limit(10):
                    message = client.messages.create(
                        model=self._model_str,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=prompt_messages[0]['content'],
                        messages=prompt_messages[1:]
                    )
                    return message.content[0].text
            except TimeoutError:
                print('Anthropic API timed out, retrying...')
            except openai.OpenAIError as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)
                
                
class VertexAIModel:
    def __init__(self, name: str, model_str: str):
        self._name = name
        self._model_str = model_str

    def name(self):
        return self._name

    def model_str(self):
        return self._model_str

    def predict(self, prompt_messages: list[dict], temperature=0, max_tokens=1000) -> str:
        client = GenerativeModel(model_name=self._model_str)
        while True:
            try:
                with time_limit(10):
                    # TODO (developer): Update and un-comment below line
                    project_id = "tinymath"

                    vertexai.init(project=project_id, location="us-central1")
                    contents = []
                    system_prompt = prompt_messages[0]['content']
                    # splice at 1 so that the system prompt isn't included
                    # this code adds the system prompt at the beginning of each user message as vertexai api does not allow system prompts (boo)
                    for i, message in enumerate(prompt_messages[1:]):
                        if message['role'] == 'user':
                            if i == 0:
                                content_text = system_prompt + "\n" + message['content']
                            else:
                                content_text = message['content']
                            contents.append(Content(parts=[Part.from_text(content_text)], role=message['role']))
                        elif message['role'] == 'assistant':
                            contents.append(Content(parts=[Part.from_text(message['content'])], role=message['role']))
                        else:
                            raise Error('Gemini models on Vertex AI only accept user and assistant messages')

                    response = client.generate_content(contents)
                    return response.text
            except TimeoutError:
                print('Vertex AI API timed out, retrying...')
            except google.api_core.exceptions.ResourceExhausted as e:
                print('Error:', e)
                print('Waiting 30s and resuming...')
                time.sleep(30)
            except ValueError as e:
                print('Error:', e)
                print('Google likely marked a problem as harmful')
                time.sleep(10)
                return "NOT AN ANSWER. GOOGLE FLAGGED THIS PROBLEM."
            except Exception as e:
                print('Error:', e)
                print('Vertex AI API timed out, retrying...')
                time.sleep(30)

class HuggingFaceModel:
    def __init__(self, name: str, model_str: str):
        self._name = name
        self._model_str = model_str
        self._tokenizer = None
        self._lm = None

    def name(self):
        return self._name

    def is_local(self) -> bool:
        return True

    def model_str(self):
        return self._model_str

    def _load_lm(self):
        if self._lm is None:
            self._lm = AutoModelForCausalLM.from_pretrained(self.model_str(),
                                                            load_in_4bit=True, device_map="auto")
            self._tokenizer = AutoTokenizer.from_pretrained(model_str)
        return self._lm

    def predict(self, prompt_messages: list[dict], temperature=0, max_tokens=300) -> str:
        prompt = format_prompt_for_completion_lm(prompt_messages)
        lm = self._load_lm()
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).cuda()

        with torch.cuda.amp.autocast():
            output = lm.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature
            )
            # Remove the prompt
            output = output[:, len(input_ids[0]):]

        detokenized = self._tokenizer.decode(output[0])

        if '###' in detokenized:
            detokenized = detokenized.split('###')[0]

        return detokenized


def remove_commas_from_numbers(input_string):
    return re.sub(r'(,\d)', lambda x: x.group(0).replace(',', ''), input_string)


def extract_first_number(input_string):
    matches = re.search(r'\b\d+(\.\d+)?(/\d+)?', input_string)
    return matches and matches.group(0)


def extract_first_sign(input_string):
    matches = re.search(r'([><=])', input_string)
    return matches and matches.group(0)


def compare_floats(answer: str, prediction: str) -> bool:
    # Compare the prediction up to the number of significant digits in it.

    if answer.count('.') == 0 or float(answer).is_integer():
        return float(prediction).is_integer() and \
            float(answer) == float(prediction)

    # Answer is a float. If prediction is an integer, it's wrong.
    if '.' not in prediction or float(prediction).is_integer():
        return False

    # Get the number of significant digits in prediction, and compare that many digits.
    a, b = answer.split('.')
    p, q = prediction.split('.')

    if a != p:
        return False

    return b.startswith(q[:-1])


def normalize(input_string: str):
    input_string = input_string.strip()
    input_string = re.sub(r'\\frac{(.*)}{(.*)}', r'\1/\2', input_string)
    input_string = re.sub(r'\\((.*)\\)', r'\1', input_string)
    input_string = re.sub(r'and', r'', input_string)
    input_string = re.sub(r'\s+', r' ', input_string)
    input_string = re.sub(r'\s/\s', r'/', input_string)
    return input_string


def extract_fraction(input_string):
    input_string = normalize(input_string)
    matches = re.search(r'((\d+)[ +\(]+)?(\d+/\d+)', input_string)

    if matches:
        integer_part = matches.group(2)
        fraction_part = matches.group(3)
        a, b = fraction_part.split('/')
        if integer_part:
            return f"{integer_part.strip()} {a.strip()}/{b.strip()}"
        else:
            return f"{a.strip()}/{b.strip()}"

    return matches and matches.group(0)


def extract_answer(generation: str, extractor) -> Optional[str]:
    matches = re.search('Answer: (.*)\\n?', generation)
    if matches:
        line = matches.groups()[0]
    else:
        line = generation.split('\n')[-1]

    line = remove_commas_from_numbers(line)
    return extractor(line)


def parse_fraction(s: str) -> F:
    """Parse a string into a fraction. Allows an optional integer part."""
    if ' ' in s:
        integer_part, fraction_part = s.split(' ')
    else:
        integer_part, fraction_part = '0', s

    return F(fraction_part) + int(integer_part)


def extract_set(extractor):
    def extract(input_string):
        a = extractor(input_string)
        if a is not None:
            _, end = input_string.split(a, 1)
            return set([a]).union(extract(end))
        return set()

    def extract_str_set(input_string):
        return ';'.join(map(str, extract(input_string)))

    return extract_str_set


def infer_answer_extractor(answer):
    if answer in "<=>":
        return extract_first_sign
    if answer.startswith('{') or answer.startswith('['):
        # Grading a set of predictions, like in a system of equations.
        first_answer = answer[1:].split(',')[0]
        return extract_set(infer_answer_extractor(first_answer))
    if '/' in answer:
        return extract_fraction
    return extract_first_number


def parse_predictions_and_answers(answers, predictions):
    predictions = predictions.split(';')
    answers = answers.split(';')
    first_answer = answers[0]
    first_prediction = predictions[0]

    if '.' in first_answer or '.' in first_prediction:
        # Floating point
        return set(float(a) for a in answers), set(float(p) for p in predictions)
    if '/' in first_answer:
        return set(parse_fraction(a) for a in answers), set(parse_fraction(p) for p in predictions)

    return set(answers), set(predictions)


def grade_prediction(answer, prediction, abstol=1e-6):
    prediction = prediction.strip()
    is_sequence = ';' in answer
    if is_sequence:
        answers, predictions = parse_predictions_and_answers(answer, prediction)
        # NOTE: this will not work for sets of floats, since it will do exact comparison,
        # but we shouldn't have that case for now.
        return answers == predictions

    if '.' in answer or '.' in prediction:
        return compare_floats(answer, prediction)
    if '/' in answer:
        if '/' in prediction:
            return parse_fraction(answer) == parse_fraction(prediction)
        # Else, convert both to floating point.
        answer_fp = float(parse_fraction(answer))
        prediction_fp = float(prediction)
        return abs(answer_fp - prediction_fp) < abstol
    return answer == prediction


def grade_reasoning(answer, model_reasoning):
    model_reasoning = normalize(model_reasoning)
    extractor = infer_answer_extractor(answer)
    try:
        model_answer = extract_answer(model_reasoning, extractor)
        if model_answer is None:
            return None
        return grade_prediction(extractor(answer), model_answer.strip())
    except ValueError:
        return None


EVAL_PROMPT = json.load(open('prompts/eval_prompt.json'))

def make_main_problem_prompt(problem):
    return EVAL_PROMPT + [
        {"role": "user", "content": problem['statement']}
    ]


def make_followup_prompt(problem, reasoning, followup):
    return EVAL_PROMPT + [
        {"role": "user", "content": problem['statement']},
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": followup['statement']}
    ]


def make_problem_lm_key(problem, lm):
    return repr((problem['id'], lm.model_str()))


def evaluate_local_lm(lm: LanguageModel, problems: list[dict], output_path: str):
    print('Evaluating', lm.name())

    try:
        with open(output_path) as f:
            results = json.load(f)
    except IOError:
        print('No results yet, creating', output_path)
        results = {}

    def save_results():
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

    sampling_params = SamplingParams(
        temperature=0.0,
        stop=MESSAGE_DELIMETER,
        max_tokens=1000,
    )
    llm = None

    # Run all the pending prompts (at the first run, we don't run follow-ups).
    while True:
        pending_prompts = {}

        # Collect prompts for all missing problems.
        for problem in problems:
            original_problem, followups = problem[0], problem[1:]

            key = make_problem_lm_key(original_problem, lm)

            if key not in results:
                pending_prompts[key] = make_main_problem_prompt(original_problem)
            elif results[key]['correct']:
                for followup in followups:
                    followup_key = make_problem_lm_key(followup, lm)
                    if followup_key not in results[key]['followups']:
                        pending_prompts[followup_key] = make_followup_prompt(
                            original_problem, results[key]['model_generation'], followup)

        if not pending_prompts:
            print('All done.')
            return

        print(len(pending_prompts), 'completions to generate.')

        keys_prompts = [(k, format_prompt_for_completion_lm(v))
                        for k, v in pending_prompts.items()]
        keys, prompts = [k for k, _ in keys_prompts], [p for _, p in keys_prompts]

        if llm is None:
            model_str_revision = lm.model_str().split(':', 1)
            if len(model_str_revision) == 2:
                model_str, revision = model_str_revision
            else:
                model_str, revision = model_str_revision[0], None
            llm = LLM(model=model_str,
                      revision=revision,
                      tensor_parallel_size=torch.cuda.device_count())

        responses = llm.generate(prompts, sampling_params)
        outputs = [r.outputs[0].text for r in responses]
        outputs_by_key = dict(zip(keys, outputs))

        with open('vllm_outputs.json', 'w') as f:
            json.dump(outputs_by_key, f)

        # Now write all prompts to 'model_generation' and grade.
        for problem in tqdm(problems):
            original_problem, followups = problem[0], problem[1:]

            key = make_problem_lm_key(original_problem, lm)

            if key not in results:
                results[key] = {
                    'problem': original_problem['id'],
                    'standard': original_problem['standard'],
                    'question': original_problem['statement'],
                    'answer': original_problem['answer'],
                    'model': lm.model_str(),
                    'model_generation': outputs_by_key[key],
                    'correct': grade_reasoning(original_problem['answer'], outputs_by_key[key]),
                    'followups': {}
                }
            elif results[key]['correct']:
                results[key]['followups'] = results[key].get('followups', {})

                for followup in followups:
                    followup_key = make_problem_lm_key(followup, lm)
                    if followup_key not in results[key]['followups']:
                        results[key]['followups'][followup_key] = {
                            'problem': followup['id'],
                            'standard': followup['standard'],
                            'question': followup['statement'],
                            'answer': followup['answer'],
                            'model': lm.model_str(),
                            'model_generation': outputs_by_key[followup_key],
                            'correct': grade_reasoning(followup['answer'], outputs_by_key[followup_key])
                        }

        save_results()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_lm(lm: LanguageModel, problems: list[dict], output_path: str):
    print('Evaluating', lm.name())

    try:
        with open(output_path) as f:
            results = json.load(f)
    except IOError:
        print('No results yet, creating', output_path)
        results = {}

    with open('prompts/eval_prompt.json') as f:
        prompt_messages = json.load(f)

    def save_results():
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

    for problem in tqdm(problems):
        original_problem, followups = problem[0], problem[1:]

        key = repr((original_problem['id'], lm.model_str()))

        if key not in results:
            generation = lm.predict(
                prompt_messages + [{"role": "user",
                                    "content": original_problem['statement']}]
            )

            answer_extractor = infer_answer_extractor(original_problem['answer'])
            try:
                lm_answer = extract_answer(generation, answer_extractor)
                correct = lm_answer and grade_reasoning(original_problem['answer'],
                                                        generation)
            except Exception as e:
                print('Error extracting the answer:', e, generation)
                lm_answer, correct = None, None

            results[key] = {
                'problem': original_problem['id'],
                'standard': original_problem['standard'],
                'question': original_problem['statement'],
                'answer': original_problem['answer'],
                'model': lm.model_str(),
                'model_generation': generation,
                'model_answer': lm_answer,
                'correct': correct,
                'followups': {}
            }
            save_results()

        problem_messages = [{"role": "user",
                             "content": original_problem['statement']},
                             {"role": "assistant",
                             "content": results[key]['model_generation']}]

        main_question_correct = results[key]['correct']
        main_question_key = key

        for followup in followups:
            key = repr((followup['id'], lm.model_str()))

            if key not in results[main_question_key]['followups'] and main_question_correct:
                generation = lm.predict(
                    prompt_messages + problem_messages +
                    [{"role": "user", "content": followup['statement']}]
                )

                answer_extractor = infer_answer_extractor(followup['answer'])
                lm_answer = extract_answer(generation, answer_extractor)
                correct = lm_answer and grade_reasoning(followup['answer'], generation)

                followup_entry = {
                    'problem': followup['id'],
                    'standard': followup['standard'],
                    'question': followup['statement'],
                    'answer': followup['answer'],
                    'model': lm.model_str(),
                    'model_generation': generation,
                    'model_answer': lm_answer,
                    'correct': correct
                }
                if 'followups' not in results[main_question_key]:
                    results[main_question_key]['followups'] = []
                results[main_question_key]['followups'][key] = followup_entry
                save_results()



def load_problems(dataset, max_per_standard=30):
    problems = []
    count_by_standard = collections.defaultdict(int)


    for filename in os.listdir(dataset):
        if not filename.endswith('.json'):
            continue

        with open(f'{dataset}/{filename}') as f:
            f_problems = json.load(f)

        for problem in f_problems:
            standard = problem[0]['standard']

            if count_by_standard[standard] == max_per_standard:
                continue

            problems.append(problem)
            count_by_standard[problem[0]['standard']] += 1

    return problems


def run_eval(
        dataset: str,
        model_names: list[str],
        max_per_standard: int,
        output_path: str,
):
    problems = load_problems(dataset, max_per_standard)

    api_lms = [
        OpenAIChatModel('GPT 3.5', 'gpt-3.5-turbo-0125'),
        OpenAIChatModel('GPT 4', 'gpt-4-turbo'),
        OpenAIChatModel('GPT 4o', 'gpt-4o-2024-05-13'),
        TogetherAIChatModel('Mistral (7B) Instruct v0.3', 'mistralai/Mistral-7B-Instruct-v0.3'),
        TogetherAIChatModel('Mixtral-8x7B Instruct (46.7B)', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
        TogetherAIChatModel('Mixtral-8x22B Instruct (141B)', 'mistralai/Mixtral-8x22B-Instruct-v0.1'),
        TogetherAIChatModel('Deepseek Coder Instruct (33B)', 'deepseek-ai/deepseek-coder-33b-instruct'),
        TogetherAIChatModel('DeepSeek LLM Chat (67B)', 'deepseek-ai/deepseek-llm-67b-chat'),
        TogetherAIModel('Microsoft Phi-2', 'microsoft/phi-2'),
        TogetherAIChatModel('Code Llama Instruct (7B)', 'codellama/CodeLlama-7b-Instruct-hf'),
        TogetherAIChatModel('Code Llama Instruct (13B)', 'codellama/CodeLlama-13b-Instruct-hf'),
        TogetherAIChatModel('Code Llama Instruct (34B)', 'codellama/CodeLlama-34b-Instruct-hf'),
        TogetherAIChatModel('LLaMA-3 Chat (8B)', 'meta-llama/Llama-3-8b-chat-hf'),
        TogetherAIChatModel('LLaMA-3 Chat (70B)', 'meta-llama/Llama-3-70b-chat-hf'),
        TogetherAIChatModel('Gemma Instruct (2B)', 'google/gemma-2b-it'),
        TogetherAIChatModel('Gemma Instruct (7B)', 'google/gemma-7b-it'),
        AnthropicModel('Claude 3 Opus', 'claude-3-opus-20240229'),
        AnthropicModel('Claude 3 Sonnet', 'claude-3-sonnet-20240229'),
        AnthropicModel('Claude 3 Haiku', 'claude-3-haiku-20240307'),
        VertexAIModel('Gemini 1.5 Pro', 'gemini-1.5-pro-001'),
        VertexAIModel('Gemini 1.5 Flash', 'gemini-1.5-flash-001'),
    ]

    evaluated_api_lms = set()

    for lm in api_lms:
        if lm.model_str() in model_names:
            print('Evluating API model', lm.name())
            evaluate_lm(lm, problems, output_path)
            evaluated_api_lms.add(lm.model_str())

    # Evaluate local models.
    for lm_name in model_names:
        if lm_name in evaluated_api_lms:
            continue

        lm = HuggingFaceModel(lm_name, lm_name)
        evaluate_local_lm(lm, problems, output_path)


def regrade(path: json):
    with open(path) as f:
        results = json.load(f)

    for k, v in results.items():
        extractor = infer_answer_extractor(v['answer'])
        model_answer = extract_answer(v['model_generation'], extractor)
        correct = model_answer and grade_reasoning(v['answer'], v['model_generation'])
        v['model_answer'] = model_answer
        v['correct'] = correct
        
        for _, v_fup in v['followups'].items():
            extractor = infer_answer_extractor(v_fup['answer'])
            model_answer = extract_answer(v_fup['model_generation'], extractor)
            correct = model_answer and grade_reasoning(v_fup['answer'], v_fup['model_generation'])
            v_fup['model_answer'] = model_answer
            v_fup['correct'] = correct

    with open(f'regraded-{path}', 'w') as f:
        json.dump(results, f, indent=4)
        print('Regraded into', f.name)


def eval_results(path, granularity='standard'):

    with open(path) as f:
        results = json.load(f)

    results = results.values()
    for r in results:
        r['grade'] = r['standard'][0]

    models = sorted(set([r['model'] for r in results]))
    standards = sorted(list(set([r['standard'] for r in results])))
    grades = sorted(list(set([r['grade'] for r in results])))

    problem_groups = []

    if granularity == 'standard':
        for s in standards:
            ids = {r['problem'] for r in results if r['standard'] == s} | {
                followup['problem'] for r in results for followup in r.get('followups', {}).values() if followup['standard'] == s
            }
            problem_groups.append((f'Standard {s}', ids))
    elif granularity == 'grade':
        for g in grades:
            # ids = {r['problem']  for r in results if r['grade'] == g}
            ids = { r['problem'] for r in results if r['grade'] == g} | {
                    followup['problem'] for r in results for followup in r.get('followups', {}).values() if r['grade'] == g
                    }
            problem_groups.append((f'Grade {g}', ids))
    else:
        ids = {r['problem'] for r in results}
        problem_groups.append(('All problems', ids))

    for problem_group, problem_ids in problem_groups:
#        print('###', problem_group)

        for m in models:
            correct = [bool(r['correct']) for r in results if r['problem'] in problem_ids and r['model'] == m and r['model_answer'] is not None]
            accuracy = 'N/A' if not correct else f'{sum(correct) / len(correct):.3f}'
            print(problem_group, ',', m, ',', accuracy)
            
            


def aggregate_results(paths, output):
    all_results = {}

    for p in paths:
        with open(p) as f:
            results_p = json.load(f)
        for k, v in results_p.items():
            all_results[k] = v
        print('Loaded', len(results_p), 'data points from', p)

    with open(output, 'w') as f:
        json.dump(all_results, f, indent=4)
        print('Wrote', output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', action='store_true', help='Run inference on LMs on dataset of problems.')
    parser.add_argument('--eval', action='store_true', help='Load results and compute accuracies')
    parser.add_argument('--regrade', action='store_true', help='Read results file, re-extract model answers and re-compute correct.')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate results files.')
    parser.add_argument('--problems-per-standard', type=int, default=30, help='Maximum number of problems per standard to run on')
    parser.add_argument('--results', type=str, nargs='*', default=['results.json'], help='Path to JSON file with results.')
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to dataset to use.')
    parser.add_argument('--results-dir', type=str, default='results', help='Path to results to write to.')
    parser.add_argument('--eval-plan', type=str, help='JSON file with the evaluation plan.')
    parser.add_argument('--model-names', type=str, default='gpt-3.5-turbo', help='Comma-separated list of model names')
    parser.add_argument('--granularity',
                        type=str,
                        default='standard',
                        choices=['standard', 'grade', 'all'],
                        help='Granularity to evaluate. standard, grade or all')

    opt = parser.parse_args()

    if opt.eval_plan:
        with open(opt.eval_plan, 'r') as f:
            eval_plan = json.load(f)

        run_eval(
            eval_plan['dataset'],
            eval_plan['model_names'],
            eval_plan['problems_per_standard'],
            os.path.join(opt.results_dir, os.path.basename(opt.eval_plan))
        )

    if opt.inference:
        run_eval(opt.dataset, opt.model_names.split(','),
                 opt.problems_per_standard, opt.results[0])
    elif opt.regrade:
        regrade(opt.results[0])
    elif opt.eval:
        eval_results(opt.results[0], opt.granularity)
    elif opt.aggregate:
        aggregate_results(opt.results[:-1], opt.results[-1])


if __name__ == '__main__':
    main()
