from plotting_scripts.styling import PAPER_COLORS


LABELS = [
    "Gemini 3.1 Pro with Mechanism Info",
    "Gemini 2.5 Pro with Mechanism Info",
    "GPT-5.4 with Mechanism Info",
    "GPT-4o with Mechanism Info",
    "Claude Opus 4.7 with Mechanism Info",
    "Claude Haiku 4.5 with Mechanism Info",
    "DeepSeek V3.2 with Mechanism Info",
    "Mistral Large 3 with Mechanism Info",
    "Llama 4 Maverick with Mechanism Info",
    "Qwen3-32B with Mechanism Info",
]

DISPLAY_LABELS = {
    "Gemini 3.1 Pro with Mechanism Info": "Gemini 3.1 Pro",
    "Claude Opus 4.7 with Mechanism Info": "Claude 4.7 Opus",
    "Claude Haiku 4.5 with Mechanism Info": "Claude 4.5 Haiku",
    "Gemini 2.5 Pro with Mechanism Info": "Gemini 2.5 Pro",
    "GPT-4o with Mechanism Info": "GPT-4o",
    "GPT-5.4 with Mechanism Info": "GPT-5.4",
    "Qwen3-32B with Mechanism Info": "Qwen3-32B",
    "DeepSeek V3.2 with Mechanism Info": "DeepSeek V3.2",
    "Llama 4 Maverick with Mechanism Info": "Llama 4 Maverick",
    "Mistral Large 3 with Mechanism Info": "Mistral Large 3",
    "Maximin LHS": "Maximin LHS",
    "Bayes": "Bayes",
    "EIG Reference": "EIG Reference",
}

MODEL_COLOR_SEQUENCE = [
    PAPER_COLORS[0],
    PAPER_COLORS[1],
    PAPER_COLORS[2],
    PAPER_COLORS[3],
    PAPER_COLORS[4],
    "#2F8F83",
    "#1F5B9A",
    "#4C78A8",
    "#8E6C8A",
    "#A86D1B",
]

MODEL_COLORS = {
    **dict(zip(LABELS, MODEL_COLOR_SEQUENCE)),
    "Maximin LHS": "#B5B5B5",
    "Bayes": "#D1D1D1",
    "EIG Reference": "#6F7F8F",
}
