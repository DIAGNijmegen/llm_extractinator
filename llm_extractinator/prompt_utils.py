from pathlib import Path

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)


def load_template(name: str) -> str:
    return (Path(__file__).parent / "prompt_templates" / f"{name}.txt").read_text()


def build_zero_shot_prompt() -> ChatPromptTemplate:
    system_text = load_template("data_extraction/system_prompt")
    human_text = load_template("data_extraction/human_prompt")
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", human_text),
        ]
    )


def build_few_shot_prompt(example_selector) -> ChatPromptTemplate:
    system_text = load_template("data_extraction/system_prompt")

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", load_template("data_extraction/human_prompt")),
            ("ai", load_template("data_extraction/ai_prompt")),
        ]
    )

    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input"],
    )

    # final human turn
    human_prompt = PromptTemplate(
        template=load_template("data_extraction/human_prompt"),
        input_variables=["input"],
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            few_shot,
            ("human", human_prompt),
        ]
    )


def build_translation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", load_template("translation/system_prompt")),
            ("human", load_template("translation/human_prompt")),
        ]
    )
