# Configuration for the GroundTruthGenerator
# This file centralizes parameters to prevent accidental misconfiguration.

GROUND_TRUTH_GENERATOR_CONFIG = {
    # Reference substrate concentration used in validation
    "s0_ref": 1.0,
    
    # Factor to determine inhibitor concentration relative to s0_ref, if an inhibitor is present.
    "inhibitor_conc_factor": 1.0,

    # Maximum simulation time for validation runs
    "max_t": 1000,
    
    # Parameters for k_vector sampling (log-uniform distribution)
    "k_sampling_params": {
        "log_min": -2,
        "log_max": 2,
    },
    
    # Whether to use the mechanism-specific condition checker
    "use_condition_checker": True,
    
    # Whether to enforce a minimum spread in the sampled k_vector values
    "make_distinguishable": False,

    # COPASI integration tuning parameters
    "copasi_config": {
        "dense_num_steps": 5000,
        "max_internal_steps": 1_000_000,
        "abs_tol": 1e-12,
        "rel_tol": 1e-8,
    },
    
    # Directory where COPASI model files (.cps) are stored
    "model_dir": "bench/models",
    
    # Directory to save the generated ground truth files
    "output_dir": "ground_truths_200_samples_v2",
}

# Configuration for KineticOracle's data generation process
KINETIC_ORACLE_DATA_CONFIG = {
    # Directory where COPASI model files (.cps) are stored
    "model_dir": "bench/models",

    # Number of data points to generate in the time course
    "num_points": 6,
    
    # Method for sampling time points (e.g., "uniform", "logspace", "power", "test_fixed")
    "time_sampling_method": "uniform",
    
    "inhibitor_conc_factor": 1.0,

    # Noise level to add to generated data (0.0 for no noise)
    "noise_level": 0.05,

    # COPASI integration tuning parameters (copied from GROUND_TRUTH_GENERATOR_CONFIG)
    "copasi_config": {
        "dense_num_steps": 5000,
        "max_internal_steps": 1_000_000,
        "abs_tol": 1e-12,
        "rel_tol": 1e-8,
    },
}

PRIOR_CONFIG = {
    'log_k_min': -2,
    'log_k_max': 2,
    'log_sigma_min': -2.5,
    'log_sigma_max': -1.0,
    'log_conc_min': -6,
    'log_conc_max': 2,
}

CMA_OPTIONS = {
    'verb_disp': 0, 
    'ftarget': -10000, 
    'tolstagnation': 20,
    'popsize_factor': 7,
    # 'maxfevals': 400, # 最大関数評価回数（多分目的関数を呼ぶ回数）
    'timeout': 60,
    'sigma0': 1.0,
    # 'tolfun': 1e-4,
    # 'tolfunrel': 1e-6,
    # 'tolx': 5e-3,
    # 'tolstagnation': 20, # 停滞許容回数
}

CMA_OPTIONS_AFTER_FIRST_RUN = CMA_OPTIONS.copy()
CMA_OPTIONS_AFTER_FIRST_RUN['sigma0'] = 0.3

RANDOM_PROPOSER_CONFIG = {
    "substrate_conc_range": [0.1, 5.0],
    "product_conc_range": [0, 0.5],
    # Epsilon for proposer-side P sampling when the lower bound is zero.
    # Chosen so that the epsilon-shifted log-uniform sampler gives
    # approximately Pr(P >= 0.1) ~= 0.25 over [0, 0.5].
    "product_sampling_epsilon": 1e-3,
    "catalyst_conc_range": [5e-4, 0.2],
    "seed": 42
}

GROUND_TRUTH_GENERATOR_CONFIG["validation_sampling"] = {
    "num_conditions": 100,
    "min_pass_fraction": 0.05,
    "product_sampling_epsilon": 1e-9,
    "substrate_conc_range": RANDOM_PROPOSER_CONFIG["substrate_conc_range"],
    "product_conc_range": RANDOM_PROPOSER_CONFIG["product_conc_range"],
    "catalyst_conc_range": RANDOM_PROPOSER_CONFIG["catalyst_conc_range"],
}

ALL_MODEL_NAMES = [
    f"M{i}" for i in range(1, 21)
]

# --- LLM and Agent Configuration ---
import os
from langchain_openai import ChatOpenAI
import dotenv
dotenv.load_dotenv()

# Specifies the model for the LLM proposer agent.
AGENT_PROPOSER_CONFIG = {
    "model_name": "gemini-2.5-pro", # The key to look up in LLM_CONFIGS
}

# Stores the base parameters needed to instantiate different LLM clients.
LLM_CONFIGS = {
    "gemini-2.5-pro": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "google/gemini-2.5-pro",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "gemini-3.1-pro": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "google/gemini-3.1-pro-preview",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "gpt-4o": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "openai/gpt-4o",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "gpt-5.4": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "openai/gpt-5.4",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "gpt-5.2-pro-reasoning-low": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "openai/gpt-5.2-pro",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
            "reasoning": {
                "effort": "low"
            }
        }
    },
    "gpt-5.2-pro-reasoning-high": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "openai/gpt-5.2-pro",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
            "reasoning": {
                "effort": "high"
            }
        }
    },
    "claude-4-5-haiku": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "anthropic/claude-haiku-4.5",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "claude-4-5-sonnet": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "anthropic/claude-sonnet-4.5",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "claude-4.7-opus": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "anthropic/claude-opus-4.7",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "qwen3-235b-instruct": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "qwen/qwen3-235b-a22b-2507",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
            }
    },
    "qwen3-32b": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "qwen/qwen3-32b",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "deepseek-r1": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "deepseek/deepseek-r1",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "llama-4-maverick": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "meta-llama/llama-4-maverick",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "mistral-large-3": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "mistralai/mistral-large-2512",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
        }
    },
    "deepseek-v3.2": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "deepseek/deepseek-v3.2",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
            "reasoning": {"enabled": True},
        }
    },
    "glm-4.7": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "z-ai/glm-4.7",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
            "reasoning": {"enabled": True} # Added based on OpenRouter example
        }
    },
    "glm-4.7-flash": {
        "class": ChatOpenAI,
        "params": {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": "z-ai/glm-4.7-flash",
            "temperature": 0,
            "streaming": False,
            "max_completion_tokens": 20000,
            "request_timeout": 120,
            "reasoning": {"enabled": True} # Added based on OpenRouter example
        }
    },
}
