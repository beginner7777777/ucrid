"""
UCRID Stage 3: LLM Judge
Few-shot prompt with top-k candidate intents + intent definitions.
Supports Anthropic Claude, OpenAI GPT-4, and local HuggingFace backends.
"""

import json
import re
import random
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Dict, List, Optional


SYSTEM_PROMPT = """\
You are a strict intent classifier for a task-oriented dialogue system.
You must choose exactly one label from the candidate intent names or output OOS.
Do not explain your answer.
Do not output reasoning traces or think tags such as <think>...</think>.
"""

PROMPT_TEMPLATE = """\
Classify the user query using only the candidate intents below.
If none of them fits, return OOS.

Rules:
- Return exactly one label.
- The label must be one of the candidate intent names or OOS.
- Output JSON only in the form {{"label": "<intent_name_or_OOS>"}}.
- Do not output analysis, reasoning, or <think> tags.

## Candidate Intents:
{candidates_block}

## Few-shot Examples:
{examples_block}

## User Query:
"{query}"

## Output JSON:
{{"label": "<intent_name_or_OOS>"}}"""


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def apply_llm_label_policy(
    llm_label: int,
    fallback_label: int,
    oos_label: int,
    policy: str = "all",
):
    """
    Decide whether to trust the LLM prediction or keep the small-model fallback.
    Returns: (resolved_label, llm_applied)
    """
    if llm_label == -1:
        return fallback_label, False

    if policy == "all":
        return llm_label, True
    if policy == "oos_only":
        if llm_label == oos_label:
            return llm_label, True
        return fallback_label, False
    if policy == "id_only":
        if llm_label != oos_label:
            return llm_label, True
        return fallback_label, False

    raise ValueError(f"Unknown LLM accept policy: {policy}")


class LLMJudge:
    """
    Stage 3 LLM judge for uncertain samples.

    Usage (local):
        judge = LLMJudge(model="/path/to/llama", backend="local")
    Usage (API):
        judge = LLMJudge(client=anthropic_client, model="claude-3-5-sonnet-20241022", backend="anthropic")
    """

    def __init__(
        self,
        client=None,                     # anthropic.Anthropic or openai.OpenAI (API backends)
        model: str = "local",
        backend: str = "local",          # "local" | "anthropic" | "openai"
        few_shot_k: int = 3,
        oos_examples: int = 2,
        max_tokens: int = 32,
        temperature: float = 0.0,
        model_path: str = None,          # path for local backend
        shuffle_candidates: bool = True,
        random_seed: Optional[int] = None,
        local_batch_size: int = 8,
        disable_thinking: bool = False,
        openai_extra_body: Optional[Dict] = None,
    ):
        self.client = client
        self.model = model
        self.backend = backend
        self.few_shot_k = few_shot_k
        self.oos_examples = oos_examples
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.shuffle_candidates = shuffle_candidates
        self._rng = random.Random(random_seed)
        self.local_batch_size = max(1, int(local_batch_size))
        self.disable_thinking = disable_thinking
        self.openai_extra_body = openai_extra_body or {}

        # Local model (lazy-loaded)
        self._local_model = None
        self._local_tokenizer = None
        self._local_device = None
        self._model_path = model_path or model

        if backend == "local":
            self._load_local_model()

    # ------------------------------------------------------------------
    # Local model loading
    # ------------------------------------------------------------------

    def _load_local_model(self):
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        print(f"Loading local LLM from: {self._model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._local_tokenizer = tokenizer

        model_config = AutoConfig.from_pretrained(self._model_path)
        if getattr(model_config, "quantization_config", None) is not None:
            model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                device_map="auto",
            )
        else:
            # Fallback for non-quantized checkpoints.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
        self._local_model = model
        self._local_device = self._infer_local_device(model)
        print("Local LLM loaded.")

    def _infer_local_device(self, model):
        # For device_map="auto", feed input to the first CUDA shard if available.
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for _, device in device_map.items():
                if isinstance(device, str) and device.startswith("cuda"):
                    return device
        return str(model.device)

    def _format_local_prompt(self, prompt: str) -> str:
        tokenizer = self._local_tokenizer
        if tokenizer is None:
            return prompt

        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            return prompt

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    def _similarity(self, query: str, example: str) -> float:
        """
        Lightweight lexical similarity for dynamic exemplar retrieval.
        """
        q_tokens = set(self._tokenize(query))
        ex_tokens = set(self._tokenize(example))

        if not q_tokens and not ex_tokens:
            return 0.0

        overlap = len(q_tokens & ex_tokens)
        union = max(len(q_tokens | ex_tokens), 1)
        jaccard = overlap / union
        seq_ratio = SequenceMatcher(None, query.lower(), example.lower()).ratio()
        contains_bonus = 0.1 if query.lower() in example.lower() or example.lower() in query.lower() else 0.0
        return jaccard + 0.35 * seq_ratio + contains_bonus

    def _select_examples(
        self,
        query: str,
        pool: List[str],
        limit: int,
    ) -> List[str]:
        if not pool or limit <= 0:
            return []

        scored = [
            (self._similarity(query, example), idx, example)
            for idx, example in enumerate(pool)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [example for _, _, example in scored[:limit]]

    def _ordered_candidates(self, intent_names: List[str]) -> List[str]:
        ordered = list(intent_names)
        if self.shuffle_candidates:
            self._rng.shuffle(ordered)
        return ordered

    def _build_candidates_block(
        self,
        intent_names: List[str],
        intent_defs: Dict[str, str],
    ) -> str:
        lines = []
        for name in intent_names:
            desc = intent_defs.get(name, name.replace("_", " "))
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _build_examples_block(
        self,
        query: str,
        intent_names: List[str],
        train_examples: Dict[str, List[str]],
        oos_pool: List[str],
    ) -> str:
        lines = []
        for name in intent_names:
            pool = train_examples.get(name, [])
            chosen = self._select_examples(query, pool, self.few_shot_k)
            for ex in chosen:
                lines.append(f'Query: "{ex}" -> {name}')
        oos_chosen = self._select_examples(query, oos_pool, self.oos_examples)
        for ex in oos_chosen:
            lines.append(f'Query: "{ex}" -> OOS')
        return "\n".join(lines)

    def build_prompt(
        self,
        query: str,
        intent_names: List[str],
        intent_defs: Dict[str, str],
        train_examples: Dict[str, List[str]],
        oos_pool: List[str],
    ) -> str:
        ordered_candidates = self._ordered_candidates(intent_names)
        candidates_block = self._build_candidates_block(ordered_candidates, intent_defs)
        examples_block = self._build_examples_block(
            query,
            ordered_candidates,
            train_examples,
            oos_pool,
        )
        return PROMPT_TEMPLATE.format(
            candidates_block=candidates_block,
            examples_block=examples_block,
            query=query,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _normalize_label(self, text: str) -> str:
        normalized = text.strip().strip('"').strip("'").strip()
        normalized = normalized.splitlines()[0]
        normalized = normalized.split(":")[-1].strip()
        normalized = normalized.rstrip(" .!?,;:")
        return normalized

    def _candidate_patterns(self, intent_names: List[str]) -> List[tuple]:
        candidates = [("OOS", "OOS")] + [(name, name) for name in intent_names]
        patterns = []
        for canonical, alias in candidates:
            patterns.append(
                (canonical, re.compile(rf"(?<![a-z0-9_]){re.escape(alias.lower())}(?![a-z0-9_])"))
            )
        return patterns

    def _match_exact_label(self, text: str, intent_names: List[str]) -> Optional[str]:
        normalized = self._normalize_label(text)
        normalized_lower = normalized.lower()
        if normalized_lower == "oos":
            return "OOS"
        for name in intent_names:
            if normalized_lower == name.lower():
                return name
        return None

    def _match_label_in_snippet(self, text: str, intent_names: List[str]) -> Optional[str]:
        lowered = text.lower()
        matches = [canonical for canonical, pattern in self._candidate_patterns(intent_names) if pattern.search(lowered)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _extract_json_label(self, raw: str) -> Optional[str]:
        cleaned = self._strip_reasoning_blocks(raw)
        raw = cleaned if cleaned else raw

        match = re.search(r'"label"\s*:\s*"([^"\n]+)"', raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
        if fenced_match:
            try:
                payload = json.loads(fenced_match.group(1))
                value = payload.get("label")
                if isinstance(value, str):
                    return value.strip()
            except json.JSONDecodeError:
                return None

        # Best-effort parse for inline JSON objects.
        for snippet in re.findall(r"\{[^{}]*\}", raw, flags=re.DOTALL):
            if '"label"' not in snippet.lower():
                continue
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            value = payload.get("label")
            if isinstance(value, str):
                return value.strip()
        return None

    def _strip_reasoning_blocks(self, raw: str) -> str:
        # Remove explicit think blocks when present.
        cleaned = re.sub(r"<think>.*?</think>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned
        return raw

    def _candidate_answer_spans(self, raw: str) -> List[str]:
        spans = []
        stripped = raw.strip()
        if stripped:
            spans.append(stripped)

        for line in raw.splitlines():
            line = line.strip()
            if line:
                spans.append(line)

        # Inspect the tail first because local models sometimes emit a header then the label.
        deduped = []
        seen = set()
        for span in list(reversed(spans)):
            if span not in seen:
                seen.add(span)
                deduped.append(span)
        return deduped

    def _parse_label(self, raw: str, intent_names: List[str]) -> str:
        raw = self._strip_reasoning_blocks(raw)
        json_label = self._extract_json_label(raw)
        if json_label:
            matched = self._match_exact_label(json_label, intent_names)
            if matched is not None:
                return matched

        spans = self._candidate_answer_spans(raw)
        for span in spans:
            matched = self._match_exact_label(span, intent_names)
            if matched is not None:
                return matched

        pattern_candidates = []
        for span in spans[:3]:
            matched = self._match_label_in_snippet(span, intent_names)
            if matched is not None:
                pattern_candidates.append(matched)
        unique_candidates = list(dict.fromkeys(pattern_candidates))
        if len(unique_candidates) == 1:
            return unique_candidates[0]

        normalized = self._normalize_label(spans[0] if spans else raw)
        normalized_lower = normalized.lower()

        best_name = None
        best_score = 0.0
        for name in ["OOS"] + intent_names:
            score = SequenceMatcher(None, normalized_lower, name.lower()).ratio()
            if score > best_score:
                best_name = name
                best_score = score
        if best_name is not None and best_score >= 0.85:
            return best_name

        return normalized

    def _call_api(self, prompt: str) -> str:
        if self.backend == "local":
            return self._call_local_batch([prompt])[0]
        elif self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        elif self.backend == "openai":
            extra_body = deepcopy(self.openai_extra_body)
            if self.disable_thinking:
                chat_kwargs = extra_body.get("chat_template_kwargs", {})
                if not isinstance(chat_kwargs, dict):
                    chat_kwargs = {}
                chat_kwargs["enable_thinking"] = False
                extra_body["chat_template_kwargs"] = chat_kwargs
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body=extra_body if extra_body else None,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                # Some vLLM deployments (e.g., Mixtral without chat template)
                # only support /v1/completions for raw prompt inference.
                msg = str(e).lower()
                if "chat template" not in msg and "transformers v4.44" not in msg:
                    raise

                fallback_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}\n"
                response = self.client.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    prompt=fallback_prompt,
                    extra_body=extra_body if extra_body else None,
                )
                text = response.choices[0].text
                return (text or "").strip()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _call_local_batch(self, prompts: List[str]) -> List[str]:
        tokenizer = self._local_tokenizer
        model = self._local_model
        if tokenizer is None or model is None:
            raise RuntimeError("Local model is not initialized.")

        formatted_prompts = [self._format_local_prompt(p) for p in prompts]
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._local_device) for k, v in inputs.items()}

        generation_config = deepcopy(model.generation_config)
        generation_config.do_sample = self.temperature > 0
        if self.temperature > 0:
            generation_config.temperature = self.temperature
        else:
            # Avoid warnings for non-sampling decoding on instruct checkpoints.
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50
        generation_config.max_length = None
        generation_config.max_new_tokens = self.max_tokens
        generation_config.pad_token_id = tokenizer.eos_token_id

        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
        )
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        outputs = []
        for i, input_len in enumerate(input_lens):
            gen_ids = output_ids[i, int(input_len):]
            outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return outputs

    def judge(
        self,
        query: str,
        intent_names: List[str],
        intent_defs: Dict[str, str],
        train_examples: Dict[str, List[str]],
        oos_pool: List[str],
        intent_name_to_id: Optional[Dict[str, int]] = None,
        oos_label: int = 150,
    ) -> Dict:
        """
        Returns:
            {
              "raw": str,          # raw LLM output
              "intent_name": str,  # predicted intent name or "OOS"
              "label": int,        # predicted label id (-1 if unrecognized)
            }
        """
        prompt = self.build_prompt(query, intent_names, intent_defs, train_examples, oos_pool)
        raw = self._call_api(prompt)

        # Parse response
        intent_name = self._parse_label(raw, intent_names)
        if intent_name.upper() == "OOS":
            label = oos_label
            intent_name = "OOS"
        elif intent_name_to_id and intent_name in intent_name_to_id:
            label = intent_name_to_id[intent_name]
        else:
            label = -1
            if intent_name_to_id:
                for k, v in intent_name_to_id.items():
                    if k.lower() == intent_name.lower():
                        label = v
                        intent_name = k
                        break

        return {"raw": raw, "intent_name": intent_name, "label": label}

    def judge_batch(
        self,
        queries: List[str],
        topk_intent_names: List[List[str]],
        intent_defs: Dict[str, str],
        train_examples: Dict[str, List[str]],
        oos_pool: List[str],
        intent_name_to_id: Optional[Dict[str, int]] = None,
        oos_label: int = 150,
    ) -> List[Dict]:
        if self.backend == "local":
            prompts = [
                self.build_prompt(q, names, intent_defs, train_examples, oos_pool)
                for q, names in zip(queries, topk_intent_names)
            ]
            results: List[Dict] = []
            for start in range(0, len(prompts), self.local_batch_size):
                end = start + self.local_batch_size
                raw_batch = self._call_local_batch(prompts[start:end])
                for raw, names in zip(raw_batch, topk_intent_names[start:end]):
                    intent_name = self._parse_label(raw, names)
                    if intent_name.upper() == "OOS":
                        label = oos_label
                        intent_name = "OOS"
                    elif intent_name_to_id and intent_name in intent_name_to_id:
                        label = intent_name_to_id[intent_name]
                    else:
                        label = -1
                        if intent_name_to_id:
                            for k, v in intent_name_to_id.items():
                                if k.lower() == intent_name.lower():
                                    label = v
                                    intent_name = k
                                    break
                    results.append({"raw": raw, "intent_name": intent_name, "label": label})
            return results

        return [
            self.judge(q, names, intent_defs, train_examples, oos_pool,
                       intent_name_to_id, oos_label)
            for q, names in zip(queries, topk_intent_names)
        ]
