try:
    import unzip_requirements
except ImportError:
    pass

import json
from typing import Dict, Any
from transformers import AutoModelWithLMHead, AutoTokenizer

LambdaDict = Dict[str, Any]


def tell_story(event: LambdaDict, context: LambdaDict) -> LambdaDict:
    """
    Function to take string input, and use text generation models to create
    a story which can serve as a natural language response to player inputs

    :param event: Input AWS Lambda event dict
    :param context: Input AWS Lambda context dict

    :return: Output AWS Lambda dict
    """
    # Decode the request
    request_body = event.get("body")
    if type(request_body) == str:
        request_body = json.loads(request_body)

    story_context = request_body["context"]
    print(story_context)

    # Load model
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")
    
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/bart-tiny-random")
    model = AutoModelWithLMHead.from_pretrained("sshleifer/bart-tiny-random")
    """
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelWithLMHead.from_pretrained("distilgpt2")

    # Compute story
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    story_input = tokenizer.encode(story_context + tokenizer.eos_token,
                                   return_tensors='pt')

    # generated a response while limiting the total chat history to 1000 tokens,
    generated_story = model.generate(story_input,
                                     pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    story_result = tokenizer.decode(generated_story[:, story_input.shape[-1]:][0],
                                    skip_special_tokens=True)

    result = {
        "statusCode": 200,
        "body": story_result,
        "headers": {"Access-Control-Allow-Origin": "*"},
    }
    return result


if __name__ == "__main__":
    response = tell_story(
        {"body": {"context": "You attack! Then, "}},
        {}
    )
    print(response)
