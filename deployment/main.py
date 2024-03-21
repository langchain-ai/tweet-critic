import concurrent.futures
import logging
import os
import random
import re

from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchainhub import Client as HubClient
from langsmith import Client
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

logging.basicConfig(level=logging.INFO)

app = App(token=os.environ["SLACK_BOT_TOKEN"])

logger = logging.getLogger(__name__)
# Configurable bits
DATASET_NAME = "Tweet Critic"
PROMPT_NAME = "wfh/tweet-critic-fewshot"
OPTIMIZER_PROMPT_NAME = "wfh/convo-optimizer"
NUM_FEWSHOTS = 15
PROMPT_UPDATE_BATCHSIZE = 5

chat_llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=1)
optimizer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=1)

## Get few-shot examples from 👍 examples
client = Client()


def _format_example(example):
    return f"""<example>
    <original>
    {example.inputs['input']}
    </original>
    <tweet>
    {example.outputs['output']}
    </tweet>
</example>"""


def few_shot_examples():
    if client.has_dataset(dataset_name=DATASET_NAME):
        # TODO: Update to randomize
        examples = list(client.list_examples(dataset_name=DATASET_NAME))
        if not examples:
            return ""
        examples = random.sample(examples, min(len(examples), NUM_FEWSHOTS))
        e_str = "\n".join([_format_example(e) for e in examples])

        return f"""

Approved Examples:
{e_str}
"""
    return ""


few_shots = few_shot_examples()


# Create the chat bot
prompt: ChatPromptTemplate = hub.pull(PROMPT_NAME)
prompt = prompt.partial(examples=few_shots)

tweet_critic = (prompt | chat_llm | StrOutputParser()).with_config(run_name="Chat Bot")


def parse_tweet(response: str):
    match = re.search(r"(.*?)<tweet>(.*?)</tweet>(.*?)", response.strip(), re.DOTALL)
    pre, tweet, post = match.groups() if match else (response, None, None)
    return pre, tweet, post


@app.event("app_mention")
def handle_mentions(event, say):
    user_input = event["text"].replace(f"<@{event['user']}>", "").strip()
    handle_message(event, say, user_input)


@app.event("message")
def handle_message_replies(event, say):
    if event.get("thread_ts"):
        user_input = event["text"].strip()
        handle_message(event, say, user_input)


def handle_message(event, say, user_input):
    original_tweet = user_input
    messages = [("user", user_input)]

    response = tweet_critic.invoke({"messages": messages})

    pre, tweet, post = parse_tweet(response)
    if pre:
        say(pre)
    if tweet is not None:
        say(
            tweet,
            metadata={
                "original_tweet": original_tweet,
                "tweet": tweet,
            },
        )
        messages.append(("assistant", response))
    if post:
        say(post)

    # Get user feedback
    say(
        "Please provide feedback on the tweet above by reacting with :thumbsup: or :thumbsdown:.",
        metadata={
            "original_tweet": original_tweet,
            "tweet": tweet,
        },
    )


@app.event("reaction_added")
def handle_reaction(body, ack, say):
    reaction = body["reaction"]
    original_tweet = body["message"]["metadata"]["original_tweet"]
    tweet = body["message"]["metadata"]["tweet"]

    if reaction in ["thumbsup", "thumbsdown"]:
        ack()
        score = 1 if reaction == "thumbsup" else 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            def create_example():
                logger.info(f"Creating example: {original_tweet} -> {tweet}")
                try:
                    client.create_example(
                        inputs={"input": original_tweet},
                        outputs={"output": tweet},
                        dataset_name=DATASET_NAME,
                    )
                    say("Approved example saved successfully.")
                except:  # noqa: E722
                    client.create_dataset(dataset_name=DATASET_NAME)
                    client.create_example(
                        inputs={"input": original_tweet},
                        outputs={"output": tweet},
                        dataset_name=DATASET_NAME,
                    )
                    say("Approved example saved successfully.")

            if score and original_tweet and tweet:
                logger.info("Saving example.")
                futures.append(executor.submit(create_example))
            else:
                logger.info("Example not saved.")

            def parse_updated_prompt(system_prompt_txt: str):
                return (
                    system_prompt_txt.split("<improved_prompt>")[1]
                    .split("</improved_prompt>")[0]
                    .strip()
                )

            def format_conversation(messages: list):
                tmpl = """<turn idx={i}>
            {role}: {txt}
            </turn idx={i}>
            """
                return "\n".join(
                    tmpl.format(i=i, role=msg[0], txt=msg[1])
                    for i, msg in enumerate(messages)
                )

            hub_client = HubClient()

            optimizer_prompt = client.pull_prompt(OPTIMIZER_PROMPT_NAME)
            optimizer = (
                optimizer_prompt
                | optimizer_llm
                | StrOutputParser()
                | parse_updated_prompt
            ).with_config(run_name="Optimizer")
            try:
                logger.info("Updating prompt.")
                conversation = format_conversation(messages)
                if score:
                    conversation = f'<rating>The conversation was rated as {reaction} by the user.</rating>\n\n{conversation}'
                updated_sys_prompt = optimizer.invoke(
                    {
                        "prompt_versions": "\n\n".join(
                            [
                                f"<prompt version={hash_}>\n{client.get_prompt_template(updated_prompt)}\n</prompt>"
                                for hash_, updated_prompt in hub_client.list_prompts(
                                    PROMPT_NAME
                                )[-PROMPT_UPDATE_BATCHSIZE:]
                            ]
                        ),
                        # current system prompt
                        "current_prompt": client.get_prompt_template(prompt),
                        "conversation": conversation,
                        "final_value": tweet,
                    }
                )
                updated_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", updated_sys_prompt),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                client.push_prompt(PROMPT_NAME, updated_prompt)
                say("Bot updated successfully!")
            except Exception as e:
                logger.warning(f"Failed to update prompt: {e}")
                pass

            concurrent.futures.wait(futures)

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()