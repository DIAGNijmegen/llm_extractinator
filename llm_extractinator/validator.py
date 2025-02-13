import random
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, ValidationError

from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.prompt_utils import prepare_fixing_prompt
from llm_extractinator.utils import extract_json_from_text


def handle_failure(annotation):
    """
    Return default/failure values depending on annotation type.
    Called only after all fix attempts failed validation.
    """
    if get_origin(annotation) is Literal:
        # If annotation is Literal[...], pick a random valid literal
        return random.choice(get_args(annotation))

    # Basic default for common types
    type_defaults = {str: "", int: 0, float: 0.0, bool: False, list: [], dict: {}}
    if annotation in type_defaults:
        return type_defaults[annotation]

    # If it's Optional[X] or Union[X, None], handle X
    if get_origin(annotation) in {Optional, Union}:
        return handle_failure(get_args(annotation)[0])

    # If annotation is a nested Pydantic BaseModel, recursively build defaults
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        nested_fields = annotation.__annotations__
        nested_data = {}
        for field_name, field_type in nested_fields.items():
            nested_data[field_name] = handle_failure(field_type)
        return annotation.model_construct(**nested_data)

    # Fallback
    return None


def validate_and_fix_results(
    results: List[Dict[str, Any]],
    parser_model: BaseModel,
    model,
    output_format: str = "json",
    max_attempts: int = 3,
    show_fixing_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Validate a list of raw LLM outputs against a given Pydantic parser_model.
    ...
    """

    # 1) Prepare a JSON parser & the "fixing" chain
    parser = JsonOutputParser(pydantic_object=parser_model)
    format_instructions = parser.get_format_instructions()

    fixing_prompt = prepare_fixing_prompt(format_instructions)
    # The chain goes: fix_prompt -> model -> StrOutputParser -> extract_json_from_text
    fixing_chain = fixing_prompt | model | StrOutputParser() | extract_json_from_text

    # 2) We'll store the entire data item plus some metadata for each record
    processed_results = []
    for r in results:
        item = dict(r)
        item["retry_count"] = 0
        item["fix_status"] = "untested"
        processed_results.append(item)

    # 3) Validate each item initially
    invalid_indices = []
    for i, item in enumerate(processed_results):
        # Convert item to dict if it's a BaseModel (unlikely but safe)
        raw_dict = item if not isinstance(item, BaseModel) else item.model_dump()
        try:
            parser_model.model_validate(raw_dict)
            processed_results[i]["fix_status"] = "success"
        except ValidationError:
            invalid_indices.append(i)

    # If none invalid initially, we're done
    if not invalid_indices:
        return processed_results

    # 4) Attempt to fix invalid items up to max_attempts times
    for attempt in range(1, max_attempts + 1):
        print(
            f"[validate_and_fix_results] Attempt {attempt}/{max_attempts}: "
            f"Fixing {len(invalid_indices)} invalid results..."
        )

        invalid_items = [processed_results[idx] for idx in invalid_indices]

        # --------------------------------------------------------
        # *** KEY FIX: only pass the relevant user text to the chain ***
        # Instead of passing the entire 'item' dictionary as str(item)
        # which can lead to 'properties' or other schema artifacts showing up.
        # --------------------------------------------------------
        chain_inputs = [
            {
                "completion": item["text"]
            }  # or item["expected_output"], etc., if relevant
            for item in invalid_items
        ]

        if show_fixing_progress:
            callbacks = [BatchCallBack(len(invalid_items))]
            chain_outputs = fixing_chain.batch(
                chain_inputs,
                config={"callbacks": callbacks},
            )
            callbacks[0].progress_bar.close()
        else:
            chain_outputs = fixing_chain.batch(chain_inputs)

        # Re-validate the chain outputs
        next_invalid_indices = []
        for local_i, chain_output in enumerate(chain_outputs):
            global_idx = invalid_indices[local_i]

            # If the chain returned a string & we expect JSON, parse
            if isinstance(chain_output, str) and output_format == "json":
                extracted = extract_json_from_text(chain_output)
                chain_output = extracted if extracted else chain_output

            # If it's a BaseModel, convert to dict
            if isinstance(chain_output, BaseModel):
                chain_output = chain_output.model_dump()

            # --------------------------------------------------------
            # *** KEY FIX: parse with Pydantic or remove unknown keys ***
            # This ensures we only keep fields that your model allows!
            # --------------------------------------------------------
            try:
                validated_data = parser_model.model_validate(chain_output).model_dump()
                # Overwrite only with validated fields
                processed_results[global_idx].update(validated_data)
                processed_results[global_idx]["retry_count"] += 1
                processed_results[global_idx]["fix_status"] = "success"

            except ValidationError:
                # Still invalid => mark for next attempt
                processed_results[global_idx]["retry_count"] += 1
                next_invalid_indices.append(global_idx)

        # If all fixed, break out
        if not next_invalid_indices:
            break

        invalid_indices = next_invalid_indices  # update for next loop

    # 5) After final attempt, any still-invalid items => handle_failure
    if invalid_indices:
        print(
            f"[validate_and_fix_results] Assigning defaults for {len(invalid_indices)} items."
        )
        for idx in invalid_indices:
            # For each field declared in parser_model, fill in default/failure values
            for field_name, field_def in parser_model.model_fields.items():
                processed_results[idx][field_name] = handle_failure(
                    field_def.annotation
                )

            processed_results[idx]["fix_status"] = "failed"

    return processed_results
