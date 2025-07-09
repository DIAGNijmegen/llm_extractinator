from pathlib import Path
from typing import List

from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)


def load_template(template_name: str) -> str:
    """
    Load a template from the 'prompt_templates' directory.
    Adjust the path according to your project's structure.
    """
    template_path = Path(__file__).parent / "prompt_templates" / f"{template_name}.txt"
    with template_path.open() as f:
        return f.read()


def build_zero_shot_prompt(
    description: str,
) -> ChatPromptTemplate:
    """
    Build a zero-shot prompt without examples.
    """
    system_template = load_template("data_extraction/system_prompt").format(
        description=description
    )
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=system_template,
            input_variables=[],
        )
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=load_template("data_extraction/human_prompt"),
            input_variables=["input"],
        )
    )

    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])


def build_few_shot_prompt(
    description: str,
    example_selector,
) -> ChatPromptTemplate:
    """
    Build a few-shot prompt with examples.
    """
    system_template = load_template("data_extraction/system_prompt").format(
        description=description
    )
    final_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=system_template,
            input_variables=[],
        )
    )

    example_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=load_template("data_extraction/human_prompt"),
            input_variables=["input"],
        )
    )

    example_ai_prompt = AIMessagePromptTemplate(
        prompt=PromptTemplate(
            template=load_template("data_extraction/ai_prompt"),
            input_variables=["output"],
        )
    )

    example_prompt = ChatPromptTemplate.from_messages(
        [example_human_prompt, example_ai_prompt]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input"],
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=load_template("data_extraction/human_prompt"),
            input_variables=["input"],
        )
    )

    return ChatPromptTemplate.from_messages(
        [final_system_prompt, few_shot_prompt, human_prompt]
    )


def build_translation_prompt() -> ChatPromptTemplate:
    """
    Prepare a prompt for translating text to English.
    """
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=load_template("translation/system_prompt"),
            input_variables=[],
        )
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=load_template("translation/human_prompt"),
            input_variables=["input"],
        )
    )

    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
