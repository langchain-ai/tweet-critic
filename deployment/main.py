import concurrent.futures
import json
import logging
import os
import random
import re
from typing import Callable
import uuid
import functools

from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchainhub import Client as HubClient
from langsmith import Client
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

logging.basicConfig(level=logging.INFO)

app = App(token=os.environ["SLACK_BOT_TOKEN"])

logger = logging.getLogger(__name__)
# Configurable bits
DATASET_NAME = "Tweet Critic"
PROMPT_NAME = "wfh/tweet-critic-fewshot"
OPTIMIZER_PROMPT_NAME = "wfh/convo-optimizer"
NUM_FEWSHOTS = 15
PROMPT_UPDATE_BATCHSIZE = 5
REQUEST_TEXT = "Please edit the tweet and submit."

# "claude-3-haiku-20240307"
chat_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=1)
optimizer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=1)

## Get few-shot examples from 👍 examples
client = Client()


@functools.lru_cache
def _get_session_id():
    return client.read_project(
        project_name=os.environ.get("LANGCHAIN_PROJECT", "default")
    ).id


@functools.lru_cache
def get_bot_user_id() -> str:
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    response = client.auth_test()
    return response["user_id"]


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

TWEET_CRITIC = (prompt | chat_llm | StrOutputParser()).with_config(run_name="Chat Bot")


def parse_tweet(response: str):
    match = re.search(r"(.*?)<tweet>(.*?)</tweet>(.*?)", response.strip(), re.DOTALL)
    pre, tweet, post = match.groups() if match else (response, None, None)
    return pre, tweet, post


@app.event("app_mention")
def handle_mentions(event, say):
    logger.info("Mentioned")
    user_input = event["text"].replace(f"<@{event['user']}>", "").strip()
    thread_ts = event.get("thread_ts") or event["ts"]
    handle_message(event, say, user_input, thread_ts)


@app.event("message")
def handle_message_events(event, say):
    logger.info("Message event")
    if event.get("subtype") == "bot_message":
        return
    if not event.get("parent_user_id"):
        # Not in a thread
        return

    if "thread_ts" in event or event.get("channel_type") == "channel":
        user_input = event["text"].strip()
        thread_ts = event.get("thread_ts") or event["ts"]
        handle_message(event, say, user_input, thread_ts)


def _get_run_url(client: Client, run_id: str):
    session_id_ = _get_session_id()
    return (
        f"{client._host_url}/o/{client._get_tenant_id()}/projects/p/{session_id_}/"
        f"r/{run_id}?poll=true"
    )


def merge_consecutive_messages(messages: list) -> list:
    if not messages:
        return []
    if len(messages) == 1:
        return messages

    result = [messages[0]]

    for current_author, current_message in messages[1:]:
        last_author, last_message = result[-1]

        if current_author == last_author:
            result[-1] = (last_author, last_message + "\n" + current_message)
        else:
            result.append((current_author, current_message))

    return result


def _get_messages(inputs_):
    thread_ts, event, user_input = (
        inputs_["thread_ts"],
        inputs_["event"],
        inputs_["user_input"],
    )
    if thread_ts:
        channel = event["channel"]
        messages = get_thread_messages(channel, thread_ts)
    else:
        messages = [("user", user_input)]
    messages = merge_consecutive_messages(messages)
    messages[-1] = (
        messages[-1][0],
        messages[-1][1] + "\n\nRemember to treat my initial message as a tweet! "
        "Respond ONLY by writing constructive criticism then "
        "writing a better tweet within <tweet></tweet> tags.",
    )
    return {"messages": messages}


def handle_message(event, say, user_input, thread_ts):
    logger.info("Handling message")
    original_tweet = user_input

    run_id = uuid.uuid4()
    response = (_get_messages | TWEET_CRITIC).invoke(
        {"thread_ts": thread_ts, "event": event, "user_input": user_input},
        {"run_id": run_id},
    )
    run_url = _get_run_url(client, str(run_id))

    pre, tweet, post = parse_tweet(response)
    txt_blocks = []
    if pre:
        txt_blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": pre},
            }
        )
    if tweet:
        context = json.dumps({"tweet": original_tweet.strip(), "thread_ts": thread_ts})
        txt_blocks.extend(
            [
                {
                    "type": "input",
                    "block_id": "tweet_input",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "tweet_input_action",
                        "initial_value": tweet.strip(),
                        "multiline": True,
                        "min_length": 50,
                    },
                    "label": {"type": "plain_text", "text": "Tweet"},
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Submit"},
                            "style": "primary",
                            "action_id": "submit_tweet",
                            "value": tweet,
                        },
                    ],
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "👍"},
                            "style": "primary",
                            "action_id": "thumbs_up",
                            "value": context,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "👎"},
                            "style": "danger",
                            "action_id": "thumbs_down",
                            "value": context,
                        },
                    ],
                },
            ]
        )
    if post:
        txt_blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": post},
            }
        )

    blocks = [
        *txt_blocks,
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"<{run_url}|:globe_with_meridians: View Run>",
                }
            ],
        },
    ]

    say(
        text=REQUEST_TEXT,
        thread_ts=thread_ts,
        metadata={
            "original_tweet": original_tweet.strip(),
        },
        blocks=blocks,
    )
    messages.append(("assistant", response))


def _get_editted_tweet(body: dict) -> str:
    keys = ["state", "values", "tweet_input", "tweet_input_action", "value"]
    val = body
    broken = False
    for key in keys:
        if key not in val or not val.get(key):
            broken = True
            break
        val = val[key]
    if broken:
        edited_tweet = str(body["actions"][0]["value"])
    else:
        edited_tweet = str(val)
    return edited_tweet


@app.action("submit_tweet")
def handle_submit_tweet(ack, body, respond):
    logger.info("Handling submitted tweet")
    ack()
    editted_tweet = _get_editted_tweet(body)
    thread_ts = body["message"]["ts"]  # Assuming this is how you get the thread_ts
    context = json.dumps({"tweet": editted_tweet, "thread_ts": thread_ts})
    respond(
        replace_original=True,
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{editted_tweet.strip()}"},
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "👍"},
                        "style": "primary",
                        "action_id": "thumbs_up",
                        "value": context,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "👎"},
                        "style": "danger",
                        "action_id": "thumbs_down",
                        "value": context,
                    },
                ],
            },
        ],
    )


@app.action("thumbs_up")
def handle_thumbs_up(ack, body, respond):
    logger.info("Handling thumbs up")
    _handle_thumbs(1, ack, body, respond)


@app.action("thumbs_down")
def handle_thumbs_down(ack, body, respond):
    logger.info("Handling thumbs down")
    _handle_thumbs(0, ack, body, respond)


def _handle_thumbs(score: int, ack: Callable, body: dict, respond: Callable):
    ack()
    context = json.loads(body["actions"][0]["value"])
    editted_tweet = context["tweet"]
    thread_ts = context["thread_ts"]
    channel_id = body["channel"]["id"]

    logger.info(f"THUMBS: {score}\n\neditted_tweet")
    respond(
        replace_original=True,
        text=editted_tweet.strip() + "\n\nThank you for your feedback!",
    )
    _process_update(score, editted_tweet, channel_id, thread_ts)


def extract_message_content(message):
    content = message.get("text", "").strip()
    if (
        content == REQUEST_TEXT
        and "metadata" in message
        and "original_tweet" in message["metadata"]
    ):
        content = message["metadata"]["original_tweet"]

    return content.strip()


def get_thread_messages(channel, thread_ts):
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    bot_user_id = get_bot_user_id()  # Retrieve the bot's user ID programmatically
    messages = []

    response = client.conversations_replies(channel=channel, ts=thread_ts)

    for message in response["messages"]:
        if message["type"] == "message":
            if message.get("user") == bot_user_id:
                messages.append(("assistant", extract_message_content(message)))
            else:
                messages.append(("user", message["text"].strip()))
        elif message["type"] == "app_mention":
            user_input = message["text"].replace(f"<@{message['user']}>", "").strip()
            messages.append(("user", user_input))

    return merge_consecutive_messages(messages)


def _process_update(score: int, tweet: str, channel: str, thread_ts: str):
    messages = get_thread_messages(channel, thread_ts)
    original_tweet = messages[0][1]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        def create_example():
            logger.info(f"Creating example: {tweet}")
            try:
                client.create_example(
                    inputs={"input": original_tweet},
                    outputs={"output": tweet},
                    dataset_name=DATASET_NAME,
                )
            except:  # noqa: E722
                client.create_dataset(dataset_name=DATASET_NAME)
                client.create_example(
                    inputs={"input": original_tweet},
                    outputs={"output": tweet},
                    dataset_name=DATASET_NAME,
                )

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

        optimizer_prompt = hub_client.pull(OPTIMIZER_PROMPT_NAME)

        def pull_prompt(hash_):
            return hub.pull(f"{PROMPT_NAME}:{hash_}")

        def get_prompt_template(prompt):
            return prompt.messages[0].prompt.template

        optimizer_prompt_future = executor.submit(hub.pull, OPTIMIZER_PROMPT_NAME)
        list_response = hub_client.list_commits(PROMPT_NAME)
        latest_commits = list_response["commits"][:PROMPT_UPDATE_BATCHSIZE]
        hashes = [commit["commit_hash"] for commit in latest_commits]
        prompt_futures = [executor.submit(pull_prompt, hash_) for hash_ in hashes]
        updated_prompts = [future.result() for future in prompt_futures]
        optimizer_prompt = optimizer_prompt_future.result()

        optimizer = (
            optimizer_prompt | optimizer_llm | StrOutputParser() | parse_updated_prompt
        ).with_config(run_name="Optimizer")
        try:
            logger.info("Updating prompt.")
            conversation = format_conversation(messages)
            if score:
                conversation = f"<rating>The conversation was rated as a score of {score}/1 by the user.</rating>\n\n{conversation}"
            updated_sys_prompt = optimizer.invoke(
                {
                    "prompt_versions": "\n\n".join(
                        [
                            f"<prompt version={hash_}>\n{get_prompt_template(updated_prompt)}\n</prompt>"
                            for hash_, updated_prompt in zip(hashes, updated_prompts)
                        ]
                    ),
                    "current_prompt": get_prompt_template(prompt),
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
            logger.info("Updating prompt")
            hub.push(PROMPT_NAME, updated_prompt)
        except Exception as e:
            logger.warning(f"Failed to update prompt: {e}")
            pass

        concurrent.futures.wait(futures)


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
