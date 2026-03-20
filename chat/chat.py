# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import argparse
import json
from datetime import datetime

from colorama import Fore, init
from gptqmodel import GPTQModel
from transformers import AutoTokenizer

init(autoreset=True)


USER_PROMPT = "User >>> \n"
ASSISTANT_PROMPT = "Assistant >>> \n"

KEY_USER = 'user'
KEY_ASSISTANT = 'assistant'

ASSISTANT_HELLO = 'How can I help you?'
EXIT_MESSAGE = 'Exiting the program.'
DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless assistant. You should think step-by-step."

DEBUG = False


def build_messages(system_prompt):
    if system_prompt:
        return [{"role": "system", "content": system_prompt}]
    return []


def load_model(model_path, trust_remote_code=False):
    print(Fore.BLUE + f"Loading model from `{model_path}` ...\n")
    model = GPTQModel.load(model_path, trust_remote_code=trust_remote_code)
    return model


def load_tokenizer(tokenizer_path, trust_remote_code=False):
    print(Fore.BLUE + f"Loading tokenizer from `{tokenizer_path}` ...\n")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)


def chat_prompt_progress(user_input, tokenizer, messages):
    user_message = {"role": KEY_USER, "content": user_input}
    messages.append(user_message)
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if DEBUG:
        debug(tokenizer, messages)
    return input_tensor


def debug(tokenizer, messages):
    print("********* DEBUG START *********")
    print("********* Chat Template info *********")
    print(tokenizer.apply_chat_template(messages, return_dict=False, tokenize=False, add_generation_prompt=True))
    print("********* DEBUG END *********")


def get_user_input():
    user_input = input(Fore.GREEN + USER_PROMPT)
    return user_input


def print_model_message(message):
    print(Fore.CYAN + f"{ASSISTANT_PROMPT}{message}\n")


def save_chat_history(chat_history, save_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"chat_history_{timestamp}.json"
    if save_path is not None:
        filename = f"{save_path}/chat_history_{timestamp}.json"
    with open(filename, 'w') as file:
        json.dump(chat_history, file, indent=4, ensure_ascii=False)
    print(Fore.YELLOW + f"Chat history saved to '{filename}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a GPT model.")
    parser.add_argument('--model_path', type=str, help="Path to the model.")
    parser.add_argument('--tokenizer_path', type=str, help="Optional tokenizer path. Useful when a quantized checkpoint should reuse the source tokenizer.")
    parser.add_argument('--save_chat_path', type=str, help="Path to save the chat history.")
    parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help='Optional system prompt. Pass an empty string to disable the system message.')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help='Maximum number of new tokens to generate per assistant turn.')
    parser.add_argument('--trust_remote_code', action='store_true', default=False,
                        help='Allow custom model/tokenizer code from Hugging Face repos.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Print Debug Info')
    args = parser.parse_args()
    if args.model_path is None:
        raise ValueError("Model path is None, Please Set `--model_path`")
    DEBUG = args.debug

    model = load_model(args.model_path, trust_remote_code=args.trust_remote_code)
    messages = build_messages(args.system_prompt)

    print(Fore.CYAN + "Welcome to GPTQModel Chat Assistant!\n")
    print(Fore.YELLOW + "You can enter questions or commands as follows:\n")
    print(Fore.YELLOW + "1. Type your question for the model.\n")
    print(Fore.YELLOW + "2. Type 'exit' to quit the program.\n")
    print(Fore.YELLOW + "3. Type 'save' to save the chat history.\n")
    print(Fore.YELLOW + f"4. Current max_new_tokens per reply: {args.max_new_tokens}\n")

    tokenizer = model.tokenizer if args.tokenizer_path is None else load_tokenizer(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    chat_history = []  # chat history

    print_model_message(ASSISTANT_HELLO)

    while True:
        user_input = get_user_input()

        if user_input.lower() == 'exit':
            print(Fore.RED + f"{EXIT_MESSAGE}\n")
            break
        elif user_input.lower() == 'save':
            save_chat_history(chat_history, args.save_chat_path)
        else:
            input_tensor = chat_prompt_progress(user_input, tokenizer, messages)
            outputs = model.generate(
                input_ids=input_tensor.to(model.device),
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
            assistant_response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

            messages.append({"role": KEY_ASSISTANT, "content": assistant_response})
            chat_history.append({KEY_USER: user_input, KEY_ASSISTANT: assistant_response})

            print_model_message(assistant_response)
