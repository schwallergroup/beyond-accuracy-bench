from matplotlib import colormaps
from matplotlib.colors import to_hex


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

# Sample plasma at equal intervals across all 13 methods. Interleave the
# comparison methods with the other models to spread its four colors apart.
COLOR_LABEL_ORDER = [
    LABELS[0],
    LABELS[1],
    LABELS[2],
    "Maximin LHS",
    LABELS[3],
    LABELS[4],
    "Bayes",
    LABELS[5],
    LABELS[6],
    "EIG Reference",
    LABELS[7],
    LABELS[8],
    LABELS[9],
]
MODEL_COLORS = {
    label: to_hex(colormaps["plasma"](0.08 + 0.84 * index / (len(COLOR_LABEL_ORDER) - 1)))
    for index, label in enumerate(COLOR_LABEL_ORDER)
}
MODEL_COLOR_SEQUENCE = [MODEL_COLORS[label] for label in LABELS]
COMPARISON_COLORS = {
    label: MODEL_COLORS[label]
    for label in (LABELS[0], "Maximin LHS", "Bayes", "EIG Reference")
}
