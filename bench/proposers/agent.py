import weave
from typing import Any, Dict, Optional
import json
import re
import openai # Import openai for APIError
from .base import Proposer, History
from .llm_models import get_proposer_llm, get_llm_response_text
from .prompts import (
    INITIAL_USER_PROMPT,
    SYSTEM_PROMPT_BOTH,
    SYSTEM_PROMPT_NLL,
    SYSTEM_PROMPT_NLL_MECHANISM_HIDDEN,
    SYSTEM_PROMPT_NLL_WITH_INSTRUCTION,
    SYSTEM_PROMPT_RAW,
)
from ..proposers.mechanisms import CANDIDATE_MECHANISMS
from ..config import RANDOM_PROPOSER_CONFIG

MAX_RETRIES = 3


class AgentProposer(Proposer):
    """
    A proposer that uses a conversational agent (LLM) to suggest the next experiment.
    History is managed externally in the main loop.
    """
    def __init__(self, model_name: str, prompt_variant: str = "standard"):
        """
        Initializes the AgentProposer.
        The LLM is configured via bench/config.py and selected by model_name.
        prompt_variant controls which system prompt family is used.
        """
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.agent_model = get_proposer_llm(model_name=self.model_name)

    @weave.op()
    def propose(
        self,
        history: History,
        proposer_name: str,
        ground_truth_name: str,
        step: int,
        num_steps: int,
        feedback_mode: str = 'nll',
        cumulative_data: Optional[Dict[str, Any]] = None,
        previous_representative_fits: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Queries the agent model with the given history to get the next experimental proposal.
        Logs LLM interactions to Weights & Biases Weave. Includes retry logic for parsing and validation.

        Args:
            history: The history of the conversation, managed by the main loop.
            proposer_name: The name of the proposer (for Weave metadata).
            ground_truth_name: The name of the ground truth (for Weave metadata).
            step: The current iteration step (for Weave metadata).
            num_steps: The total number of steps in the experiment.
            feedback_mode: The type of feedback to provide to the agent.

        Returns:
            A dictionary containing the raw response and a 'proposal' key.
        """
        print(
            f"\n--- Sending history to Agent "
            f"(Feedback Mode: {feedback_mode}, Prompt Variant: {self.prompt_variant}) ---"
        )
        
        num_proposals = num_steps

        if self.prompt_variant in {'mechanism_hidden', 'step_instruction'} and feedback_mode != 'nll':
            raise ValueError(
                f"prompt_variant={self.prompt_variant!r} is currently supported only with feedback_mode='nll'."
            )

        if feedback_mode == 'nll':
            if self.prompt_variant == 'mechanism_hidden':
                system_prompt_template = SYSTEM_PROMPT_NLL_MECHANISM_HIDDEN
            elif self.prompt_variant == 'step_instruction':
                system_prompt_template = SYSTEM_PROMPT_NLL_WITH_INSTRUCTION
            else:
                system_prompt_template = SYSTEM_PROMPT_NLL
        elif feedback_mode == 'raw_data':
            system_prompt_template = SYSTEM_PROMPT_RAW
        elif feedback_mode == 'both':
            system_prompt_template = SYSTEM_PROMPT_BOTH
        else:
            # Default to NLL if mode is unknown
            system_prompt_template = SYSTEM_PROMPT_NLL

        # Pretty-print the mechanisms for better readability in the prompt
        pretty_mechanisms = json.dumps(CANDIDATE_MECHANISMS, indent=2)

        formatted_system_prompt = system_prompt_template.format(
            num_proposals=num_proposals,
            CANDIDATE_MECHANISMS=pretty_mechanisms
        )
        
        # Always start with the system prompt
        current_history = [{"role": "system", "content": formatted_system_prompt}]

        if not history:
            print("History is empty. Adding initial user prompt.")
            current_history.append({"role": "user", "content": INITIAL_USER_PROMPT})
        else:
            # Append the existing conversation history
            current_history.extend(history)

        for attempt in range(MAX_RETRIES):
            try: # Added try-except block
                # Make the actual call to the LLM
                response = self.agent_model.invoke(current_history)
                ai_response_content = get_llm_response_text(response)
            except openai.APIError as e:
                print(f"Agent LLM API call failed (attempt {attempt+1}): {e}")
                error_message = f"LLM API call failed: {str(e)}. Please try again."
                current_history.append({"role": "assistant", "content": f"ERROR: {error_message}"})
                current_history.append({"role": "user", "content": error_message})
                continue # Skip to next attempt

            print(f"Agent raw response (attempt {attempt+1}): {ai_response_content[:500]}...")

            try:
                # 1. Parse the response
                parsed_output = self._parse_agent_response(ai_response_content)

                # 2. Validate the proposal's values
                proposal = parsed_output.get("proposal")
                if proposal:
                    s_range = RANDOM_PROPOSER_CONFIG["substrate_conc_range"]
                    p_range = RANDOM_PROPOSER_CONFIG["product_conc_range"]
                    cat_range = RANDOM_PROPOSER_CONFIG["catalyst_conc_range"]
                    
                    s_val = proposal.get("S")
                    p_val = proposal.get("P")
                    cat_val = proposal.get("cat")

                    errors = []
                    if not (s_range[0] <= s_val <= s_range[1]):
                        errors.append(f"Substrate concentration (S) of {s_val} is outside the valid range of {s_range}.")
                    if not (p_range[0] <= p_val <= p_range[1]):
                        errors.append(f"Product concentration (P) of {p_val} is outside the valid range of {p_range}.")
                    if not (cat_range[0] <= cat_val <= cat_range[1]):
                        errors.append(f"Catalyst concentration (cat) of {cat_val} is outside the valid range of {cat_range}.")

                    if errors:
                        error_message = "Your proposal is invalid. " + " ".join(errors)
                        raise ValueError(error_message)
                
                print(f"Agent parsed and validated output: {parsed_output}")
                
                # 3. If parsing and validation are successful, return the result
                return {
                    **parsed_output,
                    "raw_response": ai_response_content,
                }
            except ValueError as e:
                print(f"Attempt {attempt+1} failed: {e}")
                error_message = f"Your last response was invalid. Error: {e}. Please check the constraints and provide a valid <suggestion> tag."
                # Add AI's invalid response and the error message to the history for context
                current_history.append({"role": "assistant", "content": ai_response_content})
                current_history.append({"role": "user", "content": error_message})
        
        # If all retries fail, return an empty proposal that the main loop will handle.
        print(f"Agent failed to provide a valid proposal after {MAX_RETRIES} attempts.")
        return {"proposal": None, "raw_response": "Agent failed to provide a valid proposal."}


    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the agent's raw text response to extract a proposal.
        It looks for a <suggestion> tag and an optional <tentative_answer>.
        If not found, it raises an error.
        """
        # Try to parse a <suggestion> and <tentative_answer>
        suggestion_match = re.search(r"<suggestion>\s*(\{.*?\})\s*</suggestion>", response, re.DOTALL)
        if suggestion_match:
            try:
                json_str = suggestion_match.group(1)
                parsed_json = json.loads(json_str)

                proposal_map = {
                    "initial_concentration_of_substrate": "S",
                    "initial_concentration_of_product": "P",
                    "initial_concentration_of_catalyst": "cat",
                }
                
                proposal = {
                    proposal_map[key]: value
                    for key, value in parsed_json.items()
                    if key in proposal_map
                }

                if len(proposal) != 3:
                    raise ValueError(f"Parsed proposal does not contain all required keys. Got: {proposal.keys()}")
                
                # Also look for candidate mechanisms
                candidate_mechanisms_match = re.search(r"<candidate_mechanisms>\s*(\[.*?\])\s*</candidate_mechanisms>", response, re.DOTALL)
                candidate_mechanisms = [] # Default to empty list
                if candidate_mechanisms_match:
                    try:
                        json_str = candidate_mechanisms_match.group(1)
                        parsed_list = json.loads(json_str)
                        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                            candidate_mechanisms = parsed_list
                        else:
                            print(f"Warning: <candidate_mechanisms> content was not a JSON list of strings: {json_str}")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse <candidate_mechanisms> content as JSON list: {candidate_mechanisms_match.group(1)}")
                    except Exception as e:
                        print(f"Warning: Error processing <candidate_mechanisms>: {e}")

                return {"proposal": proposal, "candidate_mechanisms": candidate_mechanisms}

            except Exception as e:
                raise ValueError("Could not parse a valid proposal from the agent's response.") from e

        # If no suggestion is found, raise an error, which triggers the retry logic.
        raise ValueError("Could not parse a valid <suggestion> from the agent's response.")
