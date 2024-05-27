from functools import lru_cache
from importlib import resources

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from aana.core.models.chat import ChatDialog


@lru_cache(maxsize=128)
def load_chat_template(chat_template_name: str) -> str:
    """Loads a chat template from the chat templates directory.

    Args:
        chat_template_name (str): The name of the chat template to load.

    Returns:
        str: The loaded chat template.

    Raises:
        ValueError: If the chat template does not exist.
    """
    with resources.path(
        "aana.core.chat.templates", f"{chat_template_name}.jinja"
    ) as path:
        if not path.exists():
            raise ValueError(f"Chat template {chat_template_name} does not exist.")  # noqa: TRY003

        return path.read_text()


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    dialog: ChatDialog,
    chat_template_name: str | None = None,
) -> str:
    """Applies a chat template to a list of messages to generate a prompt for the model.

    If the chat template is not specified, the tokenizer's default chat template is used.
    If the chat template is specified, the template with given name loaded from the chat templates directory.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        dialog (ChatDialog): The dialog to generate a prompt for.
        chat_template_name (str, optional): The name of the chat template to use. Defaults to None, which uses the tokenizer's default chat template.

    Returns:
        str: The generated prompt.

    Raises:
        ValueError: If the tokenizer does not have a chat template.
        ValueError: If the chat template does not exist.
    """
    messages = dialog.model_dump()["messages"]

    if chat_template_name is not None:
        chat_template = load_chat_template(chat_template_name)
        tokenizer.chat_template = chat_template

    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer does not have a chat template.")  # noqa: TRY003

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
