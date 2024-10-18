import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from tqdm.auto import tqdm
from uuid import UUID
from pydantic import ValidationError
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector, BaseExampleSelector
)
from langchain_core.prompts import (
    PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate,
    AIMessagePromptTemplate, FewShotChatMessagePromptTemplate
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import VLLM
from langchain.globals import set_debug
from data_extractor.utils import preprocess_text, save_json
from data_extractor.output_parsers import load_parser

# Disable debug mode for LangChain
set_debug(False)

class RandomExampleSelector(BaseExampleSelector):
    """A custom random example selector."""
    def __init__(self, examples: List[Dict[str, Any]], k: int = 3):
        self.examples = examples
        self.k = k

    def add_example(self, example: Dict[str, Any]):
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, Any]) -> List[Dict[str, Any]]:
        return random.sample(self.examples, self.k)


class BatchCallBack(BaseCallbackHandler):
    """A callback handler to manage progress during batch processing."""
    def __init__(self, total: int):
        super().__init__()
        self.count = 0
        self.progress_bar = tqdm(total=total)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        self.count += 1
        self.progress_bar.update(1)


class Predictor:
    """
    A class responsible for generating and executing predictions on test data using a language model.
    """

    def __init__(self, model: VLLM, task_config: Dict[str, Any], examples_path: Path, num_examples: int) -> None:
        """
        Initialize the Predictor with the provided model, task configuration, and paths.
        
        Args:
            model (VLLM): The language model to use for predictions.
            task_config (Dict): Configuration for the task, including task name, input, and label details.
            examples_path (Path): Path where the generated examples are saved.
            num_examples (int): Number of examples to select for few-shot learning.
        """
        self.model = model
        self.task_config = task_config
        self.num_examples = num_examples
        self.examples_path = examples_path

        self.embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.task_name = task_config['task_name']
        self.input_name = task_config['input_name']
        self.label_name = task_config['label_name']
        self.task_info = self.load_tasks()[self.task_name]

        # Load task-specific information
        self.task = self.task_info.get('Task')
        self.type = self.task_info.get('Type')
        self.labels = self.task_info.get('Labels')
        self.length = self.task_info.get('Length')
        self.description = self.task_info.get('Description')
        self.input_field = self.task_info.get('Input_Field')

    def load_tasks(self) -> Dict[str, Any]:
        """
        Load the task information from a JSON file.
        
        Returns:
            Dict[str, Any]: A dictionary containing the task information.
        """
        task_path = Path(__file__).parent / "tasks.json"
        with task_path.open("r") as f:
            return json.load(f)

    def generate_examples(self, train_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate examples from the training data.
        
        Args:
            train_data (pd.DataFrame): The training data containing text and labels.
        
        Returns:
            List[Dict[str, Any]]: A list of generated examples with text, label, and reasoning.
        """
        print("Generating examples...")
        
        parser_model = load_parser(task_type="Example Generation", valid_items=None, list_length=None)
        self.example_parser = JsonOutputParser(pydantic_object=parser_model)
        example_format_instructions = self.example_parser.get_format_instructions()

        system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template = self._load_template("example_generation/system_prompt").format(
                    task=self.task, 
                    labels=self.labels, 
                    description=self.description
                ) + '\n**Format instructions:**\n{format_instructions}',
                input_variables=[],
                partial_variables={"format_instructions": example_format_instructions}
            )
        )
        human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("example_generation/human_prompt"), 
                input_variables=["text", "label"]
            )
        )
        example_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        # Chain: prompt -> model -> output parser
        chain = example_prompt | self.model | self.example_parser

        # Preprocess training data
        train_data_processed = [
            {"text": preprocess_text(row[self.input_name]), "label": row[self.label_name]}
            for _, row in train_data.iterrows()
        ]

        # Generate results
        callbacks = BatchCallBack(len(train_data_processed))
        results = chain.batch(train_data_processed, config={"callbacks": [callbacks]})
        reasonings = [result['reasoning'] for result in results]

        # Combine results with original data to form examples
        examples = [
            {"text": row["text"], "label": row["label"], "reasoning": reasoning}
            for reasoning, row in zip(reasonings, train_data_processed)
        ]

        # Save examples to file
        save_json(examples, outpath=self.examples_path)
        return examples

    def _create_system_prompt(self, template_base: str, format_instructions: str) -> SystemMessagePromptTemplate:
        """
        Helper function to create a system prompt template.
        
        Args:
            template_base (str): The base template string.
            format_instructions (str): Instructions for formatting output.
        
        Returns:
            SystemMessagePromptTemplate: A system message prompt template.
        """
        return SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=template_base + '\n**Format instructions:**\n{format_instructions}',
                input_variables=[],
                partial_variables={"format_instructions": format_instructions}
            )
        )

    def _create_human_prompt(self, template_name: str, input_vars: List[str]) -> HumanMessagePromptTemplate:
        """
        Helper function to create a human prompt template.
        
        Args:
            template_name (str): The name of the template file to load.
            input_vars (List[str]): List of input variables for the human prompt.
        
        Returns:
            HumanMessagePromptTemplate: A human message prompt template.
        """
        template = self._load_template(template_name)
        return HumanMessagePromptTemplate(
            prompt=PromptTemplate(template=template, input_variables=input_vars)
        )

    def _load_template(self, template_name: str) -> str:
        """
        Load a template from the 'prompt_templates' directory.
        
        Args:
            template_name (str): The name of the template file.
        
        Returns:
            str: The content of the template file.
        """
        template_path = Path(__file__).parent / "prompt_templates" / f"{template_name}.txt"
        with template_path.open() as f:
            return f.read()

    def prepare_prompt_ollama(self, examples: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Prepare the system and human prompts for few-shot learning based on provided examples.
        
        Args:
            examples (Optional[List[Dict[str, Any]]]): The examples to use for the prompt.
        """
        self.parser_model = load_parser(task_type=self.type, valid_items=self.labels, list_length=self.length)
        self.parser = JsonOutputParser(pydantic_object=self.parser_model)
        self.format_instructions = self.parser.get_format_instructions()

        if examples:
            self.example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                examples, self.embedding_model, Chroma, k=self.num_examples
            )
            self.prompt = self._build_few_shot_prompt()
        else:
            self.prompt = self._build_zero_shot_prompt()

    def _build_few_shot_prompt(self) -> ChatPromptTemplate:
        """
        Build a few-shot prompt with examples.
        
        Args:
            examples (List[Dict[str, Any]]): The few-shot examples.
        
        Returns:
            ChatPromptTemplate: A chat prompt template for few-shot learning.
        """
        final_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template = self._load_template("data_extraction/system_prompt").format(
                    task=self.task, 
                    labels=self.labels, 
                    description=self.description
                ) + '\n**Format instructions:**\n{format_instructions}',
                input_variables=[],
                partial_variables={"format_instructions": self.format_instructions}
            )
        )
        final_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/human_prompt"), 
                input_variables=["text"]
            )
        )
        example_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/human_prompt"), 
                input_variables=["text"]
            )
        )
        example_ai_prompt = AIMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/ai_prompt"), 
                input_variables=["reasoning", "label"]
            )
        )
        example_prompt = ChatPromptTemplate.from_messages([example_human_prompt, example_ai_prompt])
        final_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self.example_selector,
            input_variables=["text"]
        )
        return ChatPromptTemplate.from_messages([final_system_prompt, final_few_shot_prompt, final_human_prompt])

    def _build_zero_shot_prompt(self) -> ChatPromptTemplate:
        """
        Build a zero-shot prompt without examples.
        
        Returns:
            ChatPromptTemplate: A chat prompt template for zero-shot learning.
        """
        final_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template = self._load_template("data_extraction/system_prompt").format(
                    task=self.task, 
                    labels=self.labels, 
                    description=self.description
                ) + '\n**Format instructions:**\n{format_instructions}',
                input_variables=[],
                partial_variables={"format_instructions": self.format_instructions}
            )
        )
        final_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/human_prompt"), 
                input_variables=["text"]
            )
        )
        return ChatPromptTemplate.from_messages([final_system_prompt, final_human_prompt])

    def ollama_prepare_fixing_prompt(self) -> None:
        """Prepare a prompt for fixing incorrectly formatted JSON output."""
        system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("output_fixing/system_prompt"),
                input_variables=[]
            )
        )
        human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("output_fixing/human_prompt"),
                input_variables=["completion"],
                partial_variables={"format_instructions": self.format_instructions}
            )
        )
        self.fixing_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    def validate_and_fix_results(self, results: List[Dict[str, Any]], fixing_chain: ChatPromptTemplate, max_attempts: int = 3) -> List[Dict[str, Any]]:
        """
        Validate and batch-fix results. If any results are invalid, try fixing them in batches.
        
        Args:
            results (List[Dict[str, Any]]): The list of results to be validated and potentially fixed.
            fixing_chain (ChatPromptTemplate): The prompt chain used to attempt fixes.
            max_attempts (int, optional): Maximum number of attempts to fix each result. Defaults to 3.
        
        Returns:
            List[Dict[str, Any]]: The list of results with all items validated or attempted to be fixed.
        """
        def handle_failure(key: str) -> Any:
            """Handle failure by assigning default values for missing fields."""
            if key == 'reasoning':
                return ''
            return random.choices(self.labels, k=self.length) if self.length else random.choice(self.labels)

        # Ensure 'original_index' and 'status' are initialized for all results
        for i, result in enumerate(results):
            result['original_index'] = i
            result.setdefault('status', 'pending')

        attempt = 0
        while attempt < max_attempts:
            # Collect indices of invalid results
            invalid_indices = []
            for i, result in enumerate(results):
                try:
                    # Validate the result
                    self.parser_model.model_validate(result)
                    result['retries'] = attempt
                    result['status'] = 'success'
                except ValidationError:
                    # Add to list of invalid results
                    invalid_indices.append(i)

            # If no invalid results, exit loop
            if not invalid_indices:
                break

            # Prepare the batch for fixing and maintain index mapping
            invalid_results = [results[i] for i in invalid_indices]
            index_mapping = {i: results[i]['original_index'] for i in invalid_indices}  # Map original index
            fixing_inputs = [{"completion": str(result)} for result in invalid_results]
            print(f"Retry {attempt + 1}: Attempting to fix {len(invalid_indices)} invalid results...")

            try:
                # Attempt to fix the batch
                callbacks = BatchCallBack(len(invalid_indices))
                fixed_results = fixing_chain.batch(fixing_inputs, config={"callbacks": [callbacks]})
                callbacks.progress_bar.close()

                # Update the results with fixed outputs, using the index mapping
                for idx, fixed_result in zip(invalid_indices, fixed_results):
                    original_index = index_mapping[idx]
                    try:
                        # Validate the fixed result
                        self.parser_model.model_validate(fixed_result)
                        fixed_result['retries'] = attempt + 1
                        fixed_result['status'] = 'success'
                        fixed_result['original_index'] = original_index  # Retain original index
                        results[idx] = fixed_result
                    except ValidationError:
                        results[idx]['status'] = 'failed'
            except Exception as e:
                print(f"Batch fixing failed at attempt {attempt + 1}: {str(e)}")

            attempt += 1

        # Handle any remaining failures after max_attempts
        for i in invalid_indices:
            if results[i].get('status') != 'success':
                print(f"Failed to fix output at original index {results[i]['original_index']} after {max_attempts} attempts. Assigning default values.")
                results[i]['label'] = handle_failure('label')
                results[i]['reasoning'] = handle_failure('reasoning')
                results[i]['retries'] = max_attempts
                results[i]['status'] = 'failed'

        # Sort results back to original order using 'original_index'
        try:
            results.sort(key=lambda x: x['original_index'])  # Sort based on the original index
        except KeyError as e:
            raise KeyError(f"KeyError during sorting: {str(e)}")

        # Remove the 'original_index' key after sorting is complete
        for result in results:
            result.pop('original_index', None)

        return results

    def predict(self, test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions on the test data.
        
        Args:
            test_data (pd.DataFrame): The test data containing text.
        
        Returns:
            List[Dict[str, Any]]: A list of prediction results.
        """
        self.ollama_prepare_fixing_prompt()

        # Create a processing chain: prompt -> model -> output parser
        chain = self.prompt | self.model | self.parser
        fixing_chain = self.fixing_prompt | self.model | JsonOutputParser()

        # Preprocess test data
        test_data_processed = [{"text": preprocess_text(report)} for report in test_data[self.input_name]]
        callbacks = BatchCallBack(len(test_data_processed))
        results = chain.batch(test_data_processed, config={"callbacks": [callbacks]})
        callbacks.progress_bar.close()

        return self.validate_and_fix_results(results, fixing_chain)
