import openai
import os
import tiktoken




client = openai.OpenAI(api_key=openai.api_key)

cumulative_tokens = 0

def num_tokens_from_messages(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-4":
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += -1
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")

def chat_with_chatgpt(prompt, cumulative_tokens=0):
    """
    Chat with the 'GPT-3.5-turbo' or 'gpt-4' model.

    Parameters:
    - prompt (str): The user's input.

    Returns:
    - str: The chatbot's response.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    response_content = response.choices[0].message.content

    # Count tokens used in the API call
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_content}]
    num_tokens = num_tokens_from_messages(messages)
    cumulative_tokens += num_tokens  # Add to cumulative count

    print(f"Tokens used in API call: {num_tokens}")
    print(f"Cumulative tokens used: {cumulative_tokens}")

    return response_content, cumulative_tokens

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        response, cumulative_tokens = chat_with_chatgpt(user_input, cumulative_tokens)
        print("Chatbot: ", response)
