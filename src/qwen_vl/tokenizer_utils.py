TRAIN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def apply_chat_template_with_override(
    tokenizer,
    messages,
    chat_template: str = TRAIN_CHAT_TEMPLATE,
    **kwargs,
):
    current_template = getattr(tokenizer, "chat_template", None)
    if current_template == chat_template:
        return tokenizer.apply_chat_template(messages, **kwargs)

    tokenizer.chat_template = chat_template
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    finally:
        tokenizer.chat_template = current_template
