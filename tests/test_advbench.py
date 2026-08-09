from sterling_exploration.advbench import aggregate_concepts, build_chat_prompt_ids
from sterling_exploration.visualization import generations_html


class FakeTokenizer:
    start_header_id = 101
    end_header_id = 102
    endofchunk_token_id = 103
    eot_id = 104

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def test_chat_prompt_marks_only_user_content() -> None:
    token_ids, content = build_chat_prompt_ids(FakeTokenizer(), "ab")
    assert [token_ids[index] for index in sorted(content)] == [ord("a"), ord("b")]
    assert token_ids[-1] == ord("\n")


def test_aggregate_counts_inputs_once() -> None:
    records = [
        {
            "input_index": 7,
            "tokens": [
                {
                    "is_user_content": True,
                    "known_concepts": [
                        {"concept_id": 3, "activation": 0.8},
                        {"concept_id": 4, "activation": 0.4},
                    ],
                },
                {
                    "is_user_content": True,
                    "known_concepts": [{"concept_id": 3, "activation": 0.6}],
                },
            ],
        }
    ]
    rows = aggregate_concepts(records, concept_type="known_concepts", content_only=True)
    assert rows[0] == {
        "concept_id": 3,
        "token_firings": 2,
        "input_firings": 1,
        "mean_activation": 0.7,
        "max_activation": 0.8,
    }


def test_generations_html_colors_judgments_and_escapes_content() -> None:
    rendered = generations_html(
        [
            {
                "input_index": 0,
                "prompt": "unsafe <prompt>",
                "target": "reference",
                "response": "response",
                "judgment": {
                    "harmful_compliance": True,
                    "confidence": 0.9,
                    "rationale": "actionable",
                },
            },
            {
                "input_index": 1,
                "prompt": "second",
                "target": "reference",
                "response": "refusal",
                "judgment": {
                    "harmful_compliance": False,
                    "confidence": 0.8,
                    "rationale": "refused",
                },
            },
        ]
    )
    assert "class='card jailbreak'" in rendered
    assert "class='card not-jailbroken'" in rendered
    assert "unsafe &lt;prompt&gt;" in rendered
