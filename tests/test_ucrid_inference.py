import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")

from inference.intent_metadata import build_intent_definition, build_intent_metadata
from inference.llm_judge import LLMJudge, apply_llm_label_policy
from inference.ucrid_router import UCRIDRouter


class DummyJudge(LLMJudge):
    def __init__(self, response: str, **kwargs):
        self._response = response
        super().__init__(backend="openai", model="dummy", **kwargs)

    def _call_api(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


def test_router_routes_small_model_direct_oos_and_llm():
    router = UCRIDRouter(
        tau_accept=0.25,
        tau_reject=0.75,
        delta=0.6,
        alpha=0.5,
        top_k=2,
        oos_label=2,
    )
    router.calibrate(entropies=[0.0, 1.1], distances=[0.0, 1.0])

    logits = torch.tensor([
        [8.0, 0.1],
        [0.7, 0.7],
        [1.2, 1.2],
    ])
    hidden = torch.tensor([
        [0.02, 0.02],
        [1.6, 1.6],
        [0.25, 0.10],
    ])
    prototypes = torch.tensor([
        [0.0, 0.0],
        [0.3, 0.2],
    ])

    result = router.route(logits, hidden, prototypes)

    assert result["decisions"] == ["small_model", "direct_oos", "llm"]
    assert result["predictions"].tolist() == [0, 2, -1]
    assert len(result["topk_intents"][0]) == 2


def test_router_temperature_fitting_does_not_increase_nll():
    router = UCRIDRouter()
    logits = torch.tensor([
        [4.2, 0.1, -1.0],
        [3.4, 1.1, -0.5],
        [2.9, 0.9, -0.2],
        [-0.2, 1.8, 0.3],
    ])
    labels = torch.tensor([1, 1, 0, 2])

    before = F.cross_entropy(router.scale_logits(logits), labels).item()
    router.fit_temperature(logits, labels, max_iter=25)
    after = F.cross_entropy(router.scale_logits(logits), labels).item()

    assert router.temperature > 0
    assert after <= before + 1e-6


def test_llm_judge_build_prompt_uses_similarity_ranked_examples():
    judge = DummyJudge(
        response="balance",
        few_shot_k=2,
        oos_examples=1,
        shuffle_candidates=False,
        random_seed=7,
    )

    prompt = judge.build_prompt(
        query="check the balance on my card",
        intent_names=["balance", "transfer"],
        intent_defs={"balance": "Balance intent", "transfer": "Transfer intent"},
        train_examples={
            "balance": [
                "check my card balance",
                "what is my account balance",
                "book a taxi",
            ],
            "transfer": [
                "transfer money to savings",
                "send cash to a friend",
            ],
        },
        oos_pool=["what is the weather today", "play some jazz"],
    )

    assert 'Query: "check my card balance" -> balance' in prompt
    assert 'Query: "what is my account balance" -> balance' in prompt
    assert 'Query: "book a taxi" -> balance' not in prompt


def test_llm_judge_parses_non_exact_response():
    judge = DummyJudge(
        response="The best label is: transfer",
        shuffle_candidates=False,
    )

    result = judge.judge(
        query="send money to savings",
        intent_names=["balance", "transfer"],
        intent_defs={"balance": "Balance", "transfer": "Transfer"},
        train_examples={"balance": [], "transfer": []},
        oos_pool=[],
        intent_name_to_id={"balance": 0, "transfer": 1},
        oos_label=150,
    )

    assert result["intent_name"] == "transfer"
    assert result["label"] == 1


def test_llm_judge_parses_json_and_avoids_prompt_echo_false_match():
    judge = DummyJudge(
        response='{"label": "transfer"}',
        shuffle_candidates=False,
    )

    result = judge.judge(
        query="send money to savings",
        intent_names=["balance", "transfer"],
        intent_defs={"balance": "Balance", "transfer": "Transfer"},
        train_examples={"balance": [], "transfer": []},
        oos_pool=[],
        intent_name_to_id={"balance": 0, "transfer": 1},
        oos_label=150,
    )

    assert result["intent_name"] == "transfer"
    assert result["label"] == 1


def test_llm_judge_prefers_trailing_label_when_model_emits_header_then_answer():
    judge = DummyJudge(
        response="intent_name_or_OOS\ntransfer",
        shuffle_candidates=False,
    )

    result = judge.judge(
        query="send money to savings",
        intent_names=["balance", "transfer"],
        intent_defs={"balance": "Balance", "transfer": "Transfer"},
        train_examples={"balance": [], "transfer": []},
        oos_pool=[],
        intent_name_to_id={"balance": 0, "transfer": 1},
        oos_label=150,
    )

    assert result["intent_name"] == "transfer"
    assert result["label"] == 1


def test_llm_judge_formats_local_prompt_with_chat_template():
    judge = DummyJudge(response="balance", shuffle_candidates=False)

    class FakeTokenizer:
        chat_template = "enabled"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            return "<chat>" + messages[1]["content"] + "</chat>"

    judge._local_tokenizer = FakeTokenizer()
    formatted = judge._format_local_prompt("prompt body")

    assert formatted == "<chat>prompt body</chat>"


def test_apply_llm_label_policy_supports_conservative_oos_only_mode():
    assert apply_llm_label_policy(150, 3, 150, policy="oos_only") == (150, True)
    assert apply_llm_label_policy(7, 3, 150, policy="oos_only") == (3, False)
    assert apply_llm_label_policy(7, 3, 150, policy="all") == (7, True)
    assert apply_llm_label_policy(7, 3, 150, policy="id_only") == (7, True)
    assert apply_llm_label_policy(150, 3, 150, policy="id_only") == (3, False)


def test_intent_metadata_builds_definition_and_maps():
    class Example:
        def __init__(self, text, label, intent_name, is_oos):
            self.text = text
            self.label = label
            self.intent_name = intent_name
            self.is_oos = is_oos

    examples = [
        Example("check my card balance", 0, "balance", False),
        Example("what is my balance", 0, "balance", False),
        Example("transfer money to savings", 1, "transfer", False),
        Example("what is the weather", 150, "oos", True),
    ]

    intent_names, intent_defs, train_examples, oos_pool, intent_name_to_id = build_intent_metadata(
        examples,
        num_intents=2,
        oos_label=150,
    )

    assert intent_names == ["balance", "transfer"]
    assert "balance" in intent_defs
    assert "balance" in build_intent_definition("balance", train_examples["balance"]).lower()
    assert oos_pool == ["what is the weather"]
    assert intent_name_to_id["transfer"] == 1
