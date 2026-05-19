# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import importlib.util
import os
import re
import json
import string
import sys
from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from openai import APIConnectionError, OpenAI, RateLimitError

import datasets
from model import InstructModel
from prompts import IN_CONTEXT_EXAMPLES, INSTRUCTIONS, INSTRUCTIONS_REASONING, IN_CONTEXT_EXAMPLES_REASONING


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_system_message(type='outcome'):
    if type == 'outcome':
        return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES
    elif type == 'reasoning':
        return INSTRUCTIONS_REASONING + "\n" + IN_CONTEXT_EXAMPLES_REASONING
    else:
        raise ValueError(f"Invalid type: {type}")


def parse_response(response: str):
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
        
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1


def load_medical_answer_parser():
    repo_root = Path(__file__).resolve().parents[1]
    parser_path = repo_root / "training" / "verl" / "verl" / "utils" / "reward_score" / "medical_answer_parser.py"
    spec = importlib.util.spec_from_file_location("medical_answer_parser", str(parser_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load medical answer parser from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_data_in_batches(data, batch_size, n_sample=None):

    def initialize_batch():
        return {
            "interaction_id": [], 
            "query": [], 
            "query_time": [], 
            "answer": [], 
            "alt_ans": [],
            "domain": [], 
            "question_type": [], 
            "static_or_dynamic": [],
            "retrieved_chunks": []
            }

    try:

        if n_sample:
            data = data.select(range(0, n_sample))

        print(f">>>>Total number of questions: {len(data)}")
        
        batch = initialize_batch()
        for item in data:
            try:
                for key in batch:
                    batch[key].append(item[key])

                if len(batch["query"]) == batch_size:
                    yield batch
                    batch = initialize_batch()
            except json.JSONDecodeError:
                logger.warn("Warning: Failed to decode a line.")

        if batch["query"]:
            yield batch
    except Exception as e:
        logger.error(f"Error: An error occurred while reading the data {e}")
        raise e

def generate_predictions(dataset, participant_model, save_path=None, n_sample=None, top_p=0.9, temperature=0.6, n_answer=32, max_new_tokens=2048, max_seq_length=16384, is_rag=True):

    ids, queries, query_times, prompt_lengths, ground_truths, alternative_answers, predictions, domains, question_types, static_or_dynamics, retrieved_chunks = [], [], [], [], [], [], [], [], [], [], []
    prompts = []
    batch_size = participant_model.get_batch_size()

    for batch in tqdm(load_data_in_batches(dataset, batch_size, n_sample), desc="Generating predictions"):
        batch_ground_truths = batch.pop("answer")
        batch_formatted_prompts, batch_prompt_lengths, batch_predictions = participant_model.batch_generate_answer(batch, top_p=top_p, temperature=temperature, n_answer=n_answer, max_new_tokens=max_new_tokens, max_seq_length=max_seq_length, is_rag=is_rag)
        
        prompts.extend(batch_formatted_prompts)
        ids.extend(batch["interaction_id"])
        queries.extend(batch["query"])
        query_times.extend(batch["query_time"])
        prompt_lengths.extend(batch_prompt_lengths)
        ground_truths.extend(batch_ground_truths)
        alternative_answers.extend(batch["alt_ans"])
        domains.extend(batch["domain"])
        question_types.extend(batch["question_type"])
        static_or_dynamics.extend(batch["static_or_dynamic"])
        predictions.extend(batch_predictions)
        retrieved_chunks.extend(batch["retrieved_chunks"])
        
    max_prompt_length = max(prompt_lengths)
    print(f">>>>Max prompt length out of {len(prompt_lengths)} prompts: {max_prompt_length}")
    print(f'>>> How many predictions: {len(predictions)}')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving predictions to {save_path}")
        new_data = []
        for id, query, query_time, prompt_length, ground_truth, alternative_answers, prompt, prediction, domain, question_type, static_or_dynamic, retrieved_chunks in zip(ids, queries, query_times, prompt_lengths, ground_truths, alternative_answers, prompts, predictions, domains, question_types, static_or_dynamics, retrieved_chunks):
            new_data.append({
                "interaction_id": id,
                "query": query,
                "query_time": str(query_time),
                "ground_truth": ground_truth,
                "alternative_answers": alternative_answers,
                "prediction": prediction,
                "llm_as_a_judge": None,
                "domain": domain,
                "question_type": question_type,
                "static_or_dynamic": static_or_dynamic,
                "prompt": prompt,
                "prompt_length": prompt_length,
                "retrieved_chunks": retrieved_chunks
            })

        print(f'>>> How many new data: {len(new_data)}')
        with open(save_path, "w") as f:
            json.dump(new_data, f, indent=4)
        logger.info(f"Predictions saved to {save_path}")
    
    return queries, ground_truths, alternative_answers, prompts, predictions



def evaluate_predictions(queries, ground_truths, alt_answers, predictions, evaluation_model_name, temperature=0, top_p=0.9, max_new_tokens=512, base_url="http://localhost:8000/v1"):
    
    n_miss, n_correct, n_exact_match = 0, 0, 0
    n_no_boxed = 0
    system_message = get_system_message(type='outcome')
    llm_responses = []
    answer_parser = load_medical_answer_parser()
    
    client = OpenAI(
        base_url=base_url,
        api_key="token-abc123",
    )
    
    def attempt_OpenAI_api_call(messages, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=evaluation_model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content
            except (APIConnectionError, RateLimitError) as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying... ({attempt + 2}/{max_retries})")
                else:
                    print(f"All {max_retries} attempts failed. Last error: {e}")
                    return None
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None
    
    for _idx, prediction_list in enumerate(tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    )):
        query = queries[_idx]
        ground_truth = ground_truths[_idx]
        alt_ans = alt_answers[_idx]

        if alt_ans is None or len(alt_ans) == 0 or str(alt_ans) == "[]":
            alt_ans_str = "[]"
        else:
            alt_ans_str = str(alt_ans)
        if not alt_ans_str.startswith("["):
            alt_ans_str = f"['{alt_ans_str}']"
        alt_answer = ast.literal_eval(alt_ans_str)

        alt_answer = [str(ans) for ans in alt_answer]

        gts = [ground_truth] + alt_answer
        
        best_score = -100
        best_eval_explanation = None
        best_prediction = None
        query_eval_results = []
        
        for prediction in tqdm(prediction_list, total=len(prediction_list), desc=f"Evaluating Pass@{len(prediction_list)} Predictions"):
            prediction = prediction.strip().lower()

            is_correct = False 
            is_exact_match = False
            eval_explanation = None
            
            # Preserve legacy eval semantics: use the first global boxed answer.
            parsed_prediction = answer_parser.parse_medical_answer(
                prediction,
                policy=answer_parser.POLICY_FIRST_BOXED_LEGACY,
            )

            if parsed_prediction.parse_status != "no_boxed":
                prediction = parsed_prediction.boxed_raw.strip()
            else:
                eval_explanation = {"score": -1, "explanation": "Evaluation Error: prediction not in \\boxed{} format"}
                query_eval_results.append(eval_explanation)

                if eval_explanation and eval_explanation["score"] > best_score:
                    best_score = eval_explanation["score"]
                    best_eval_explanation = eval_explanation

                continue
            
            for gt in gts:
                gt = str(gt).lower()
                if "i dont know" in normalize_answer(prediction):
                    eval_explanation = {"score": 0, "explanation": "missing (no llm-as-a-judge applied)"}
                    break
                elif normalize_answer(prediction) == normalize_answer(gt):
                    is_correct = True
                    is_exact_match = True
                    eval_explanation = {"score": 1, "explanation": "exact match"}
                    break
                elif "invalid question" in normalize_answer(prediction) and "invalid question" not in normalize_answer(gt):
                    eval_explanation = {"score": -1, "explanation": "the question is valid but prediction says the question is invalid"}
                else:
                    messages = [
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": f"Question: {query}\n Ground truth: {gt}\n Prediction: {prediction}\n",
                        },
                    ]
                    
                    llm_response = attempt_OpenAI_api_call(messages)
                    if llm_response:
                        parsed_response, is_correct = parse_response(llm_response)
                        eval_explanation = {"score": 1 if is_correct == 1 else -1, "explanation": parsed_response}
                        if is_correct == 1:
                            break
                    else:
                        eval_explanation = {"score": -1, "explanation": "Evaluation Error: API call failed"}

            if eval_explanation:
                query_eval_results.append(eval_explanation)
            else:
                query_eval_results.append({"score": -1, "explanation": "Evaluation Error: API call failed"})

            if eval_explanation and eval_explanation["score"] > best_score:
                best_score = eval_explanation["score"]
                best_eval_explanation = eval_explanation
                best_prediction = prediction

        llm_responses.append(query_eval_results)
        
        if best_eval_explanation:
            if best_eval_explanation["explanation"] == "missing (no llm-as-a-judge applied)":
                n_miss += 1
            elif best_eval_explanation["explanation"] == "Evaluation Error: prediction not in \\boxed{} format":
                n_no_boxed += 1
        else:
            pass

        if best_score == 1:
            n_correct += 1
        if best_eval_explanation and best_eval_explanation["explanation"] == "exact match":
            n_exact_match += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss - n) / n,
        "accuracy": n_correct / n,
        "em": n_exact_match / n,
        "hallucination (include no boxed)": (n - n_correct - n_miss) / n,
        "hallucination (exclude no boxed)": (n - n_correct - n_miss - n_no_boxed) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_exact_match": n_exact_match,
        "n_hallucination (include no boxed)": n - n_correct - n_miss,
        "n_hallucination (exclude no boxed)": n - n_correct - n_miss - n_no_boxed,
        "n_no_boxed": n_no_boxed,
        "n_no_boxed_percentage": n_no_boxed / n,
        "total": n,
    }
    logger.info(results)

    return results, llm_responses


def generate_results(generator, generator_name, TOP_K, n_sample=None, split='test', prefix=None, top_p=0.9, temperature=0.6, n_answer=64, max_new_tokens=2048, max_seq_length=16384, dataset_name='CRAG'):

    is_rag = True if TOP_K > 0 else False

    if prefix == 'Greedy':
        assert temperature == 0, "Greedy should have temperature 0"
        assert n_answer == 1, "Greedy should have n_answer 1"
    elif prefix == 'Sampling':
        assert temperature > 0, "Sampling should have temperature > 0"
        assert n_answer > 1, "Sampling should have n_answer > 1"

    dataset = datasets.load_dataset(f'weizhepei/TruthRL-{dataset_name}', split=split)
    
    METHOD = 'RAG'
    PREFIX = f"{prefix}_{METHOD}" if prefix else METHOD

    # Generate predictions
    output_path = f"results/{dataset_name}/{split}/{generator_name.split('/')[-1]}/{PREFIX}/{n_answer}_responses/results_top_{TOP_K}.json"

    generate_predictions(dataset, generator, save_path=output_path, n_sample=n_sample, top_p=top_p, temperature=temperature, n_answer=n_answer, max_new_tokens=max_new_tokens, max_seq_length=max_seq_length, is_rag=is_rag)


def compute_metrics(generator_name, TOP_K, split='test', prefix=None, n_answer=1, llm_judge="llama-70b", temperature=0, top_p=0.9, max_new_tokens=512, dataset_name='CRAG', n_sample=None, base_url="http://localhost:8000/v1"):

    METHOD = 'RAG'
    PREFIX = f"{prefix}_{METHOD}" if prefix else METHOD

    results_path = f"results/{dataset_name}/{split}/{generator_name.split('/')[-1]}/{PREFIX}/{n_answer}_responses/results_top_{TOP_K}.json"
    with open(results_path, "r") as f:
        results = json.load(f)

    if n_sample is not None:
        results = results[:n_sample]

    queries = [item["query"] for item in results]
    ground_truths = [item["ground_truth"] for item in results]
    alt_answers = [item["alternative_answers"] for item in results]
    predictions = [item["prediction"] for item in results]

    evaluation_results, llm_responses = evaluate_predictions(queries, ground_truths, alt_answers, predictions, llm_judge, temperature, top_p, max_new_tokens, base_url=base_url)

    output_dir = f"results/{dataset_name}/{split}/{generator_name.split('/')[-1]}/{PREFIX}/{n_answer}_responses/judge_{llm_judge.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/metrics_top_{TOP_K}.json", "w") as f:
        json.dump({"metrics": evaluation_results}, f, indent=4)
    
    assert len(results) == len(llm_responses), f"The number of results ({len(results)}) and llm_responses ({len(llm_responses)}) should be the same"
    
    for item, llm_response in zip(results, llm_responses):
        item["llm_as_a_judge"] = llm_response
        for llm_response_item in llm_response:
            llm_response_item.update({"judge_model": llm_judge})

    with open(f"{output_dir}/results_top_{TOP_K}.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f'>>>> Evaluation results {output_dir}/metrics_top_{TOP_K}.json:')
    print(evaluation_results)

if __name__ == "__main__":

    dataset2testsplit = {
        'CRAG': 'test',
        'NaturalQuestions': 'test',
        'HotpotQA': 'validation', # dev
        'MuSiQue': 'validation'} # dev

    api_url="http://localhost:8000/v1"

    for generator_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
        model = InstructModel(model_name=generator_name, decode_batch_size=4, vllm_tensor_parallel_size=4, vllm_gpu_memory_utilization=0.85)
        for dataset in dataset2testsplit:
            for prefix in ['Greedy']:
                for split in [dataset2testsplit[dataset]]:
                    for TOP_K in [50]:
                        for n_answer in [1]:
                            for ctx_length in [32768]:
                                print(f">>>>> Generating results for {dataset}/{split}/{generator_name}/{prefix}/{n_answer}_responses/results_top_{TOP_K}.json")
                                generate_results(model, generator_name, TOP_K, prefix=prefix, n_sample=None, temperature=0, top_p=0.9, n_answer=n_answer, max_new_tokens=2048, max_seq_length=ctx_length, split=split, dataset_name=dataset)
                                for llm_judge in ["meta-llama/Llama-3.3-70B-Instruct"]:
                                    compute_metrics(generator_name, TOP_K, prefix=prefix, n_answer=n_answer, llm_judge=llm_judge, temperature=0, max_new_tokens=512, dataset_name=dataset, n_sample=None, split=split, base_url=api_url)  