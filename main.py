import functools
from typing import Optional, cast
import uuid
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langsmith import Client
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
import re
import random
from langchain import hub
from langchainhub import Client as HubClient
import logging
import concurrent.futures

logging.basicConfig(level=logging.INFO)
st.set_page_config(
    page_title="Prompt Optimization with Feedback",
    page_icon="🦜️️🛠️",
)


logger = logging.getLogger(__name__)
# Configurable bits
DATASET_NAME = "Tweet Critic"
PROMPT_NAME = "wfh/tweet-critic-fewshot"
OPTIMIZER_PROMPT_NAME = "wfh/convo-optimizer"
NUM_FEWSHOTS = 15
PROMPT_UPDATE_BATCHSIZE = 5

chat_llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=1)
optimizer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=1)

# Add a sidebar
st.sidebar.title("Session Information")
st.sidebar.text(
    """Below are the prompts used for the
chat bot and "optimizer"
respectively.

The optimizer updates the LLM's
prompt at the end of the session
based on the conversation flow
and final edited tweet."""
)
prompt_version = st.sidebar.text_input("Prompt Version", value="latest")
prompt_url = f"https://smith.langchain.com/hub/{PROMPT_NAME}"
if prompt_version and prompt_version != "latest":
    prompt_url = f"{prompt_url}/{prompt_version}"
st.sidebar.markdown(f"[See Prompt in Hub]({prompt_url})")
optimizer_prompt_url = f"https://smith.langchain.com/hub/{OPTIMIZER_PROMPT_NAME}"
st.sidebar.markdown(f"[See Optimizer Prompt in Hub]({optimizer_prompt_url})")

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


if st.session_state.get("few_shots"):
    few_shots = st.session_state.get("few_shots")
else:
    few_shots = few_shot_examples()
    st.session_state["few_shots"] = few_shots


# Create the chat bot

prompt: ChatPromptTemplate = hub.pull(
    PROMPT_NAME
    + (f":{prompt_version}" if prompt_version and prompt_version != "latest" else "")
)

prompt = prompt.partial(examples=few_shots)


tweet_critic = (prompt | chat_llm | StrOutputParser()).with_config(run_name="Chat Bot")


def parse_tweet(response: str, turn: int, box=None):
    match = re.search(r"(.*?)<tweet>(.*?)</tweet>(.*?)", response.strip(), re.DOTALL)
    box = box or st
    pre, tweet, post = match.groups() if match else (response, None, None)
    if pre:
        box.markdown(pre)
    if tweet is not None:
        tweet = st.text_area(
            "Edit this to save your refined tweet.",
            tweet.strip(),
            key=f"tweet_{turn}",
            height=500,
        )
    if post:
        box.markdown(post)
    return tweet


def log_feedback(
    value: dict,
    *args,
    presigned_url: str,
    original_tweet: Optional[str] = None,
    txt: Optional[str] = None,
    **kwargs,
):
    st.session_state["session_ended"] = True
    st.write(
        "Thank you for your feedback! We are updating our bot based on your input."
    )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        score = {"👍": 1, "👎": 0}.get(value["score"]) or 0
        comment = value.get("text")
        futures.append(
            executor.submit(
                client.create_feedback_from_token,
                presigned_url,
                score=int(score),
                comment=comment,
            )
        )

        def create_example():
            # If the input/output pairs are provided, you can log them to a few-shot dataset.
            logger.info(f"Creating example: {original_tweet} -> {txt}")
            try:
                client.create_example(
                    inputs={"input": original_tweet},
                    outputs={"output": txt},
                    dataset_name=DATASET_NAME,
                )
                st.write("Approved example saved successfully.")
            except:  # noqa: E722
                client.create_dataset(dataset_name=DATASET_NAME)
                client.create_example(
                    inputs={"input": original_tweet},
                    outputs={"output": txt},
                    dataset_name=DATASET_NAME,
                )
                st.write("Approved example saved successfully.")

        if score and original_tweet and txt:
            logger.info("Saving example.")
            futures.append(executor.submit(create_example))
        else:
            logger.info("Example not saved.")
        with st.spinner("Updating bot's instructions..."):

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

            # Generate a new prompt. This is long..

            hub_client = HubClient()

            def pull_prompt(hash_):
                return hub.pull(f"{PROMPT_NAME}:{hash_}")

            def get_prompt_template(prompt):
                return cast(
                    SystemMessagePromptTemplate, prompt.messages[0]
                ).prompt.template

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
            conversation = format_conversation(
                st.session_state.get("langchain_messages", [])
            )
            if score:
                conversation = f'<rating>The conversation was rated as {value["score"]} by the user.</rating>\n\n{conversation}'
            updated_sys_prompt = optimizer.invoke(
                {
                    "prompt_versions": "\n\n".join(
                        [
                            f"<prompt version={hash_}>\n{get_prompt_template(updated_prompt)}\n</prompt>"
                            for hash_, updated_prompt in zip(hashes, updated_prompts)
                        ]
                    ),
                    # current system prompt
                    "current_prompt": get_prompt_template(prompt),
                    "conversation": conversation,
                    "final_value": txt,
                }
            )
            updated_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", updated_sys_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            hub.push(PROMPT_NAME, updated_prompt)
            st.success("Bot updated successfully!")
        except Exception as e:
            logger.warning(f"Failed to update prompt: {e}")
            pass

        concurrent.futures.wait(futures)


messages = st.session_state.get("langchain_messages", [])
original_tweet = messages[0][1] if messages else None
for i, msg in enumerate(messages):
    with st.chat_message(msg[0]):
        if i == len(messages) - 1 and len(msg) == 3:
            updated = parse_tweet(msg[1], i)
            presigned_url = msg[2]
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                on_submit=functools.partial(
                    log_feedback,
                    presigned_url=presigned_url,
                    original_tweet=original_tweet,
                    txt=updated,
                ),
                key=f"fb_{i}",
            )
        else:
            updated = None
            st.markdown(msg[1])
            presigned_url = None


# Run the chat conversation
run_id = uuid.uuid4()
presigned = client.create_presigned_feedback_token(
    run_id, feedback_key="tweet_critique_quality"
)
if st.session_state.get("session_ended"):
    st.write(
        "Thanks for the feedback! This session has ended, copy the final tweet above"
    )
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()
else:
    if not messages:
        st.write("Paste your initial tweet in the text box below.")
    if user_input := st.chat_input(placeholder="Write here..."):
        st.chat_message("user").write(user_input)
        original_tweet = user_input
        messages.append(("user", user_input))
        with st.chat_message("assistant"):
            write_stream = tweet_critic.stream(
                {"messages": [tuple(msg[:2]) for msg in messages]},
                config={"run_id": run_id},
            )
            message_placeholder = st.empty()
            full_response = ""
            for chunk in write_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown("")
            tweet_txt = parse_tweet(full_response, len(messages), message_placeholder)
            messages.append(("assistant", full_response, presigned.url))
        st.session_state["langchain_messages"] = messages
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            on_submit=functools.partial(
                log_feedback,
                presigned_url=presigned.url,
                original_tweet=original_tweet,
                txt=tweet_txt,
            ),
            key=f"fb_{len(messages) - 1}",
        )
