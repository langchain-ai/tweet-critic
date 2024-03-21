# Online Prompt Optimization

<!-- markdown-link-check-disable -->
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/langchain-ai/tweet-critic)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tweet-critic.streamlit.app/)
<!-- markdown-link-check-enable -->

Welcome to the tweet critic prompt optimization demo! This repository showcases a Streamlit application that demonstrates how to iteratively refine and optimize prompts based on user feedback and dialog.

Have you ever found yourself having the _same_ conversation with ChatGPT, Claude, or similar application? It can often take many interactions to get get it to generate a useful response with the desired voice, tone and other characteristics. This demo shows how to make system updates based on user dialog to compile out this feedback so that subsequent interactions are less onerous.

It employs a two common optimization techniques:

- Few-shot Learning: approved examples (those given a 👍) are added to a LangSmith few-shot prompt dataset and later dynamically injected in the chat bot's system prompt.
- Prompt Optimization: When a 👍 or 👎 is provided, an "optimizer" prompt is used to propose updates to the chat bot to help it better satisfy this type of user request in the future. These updates are checked in the the [LangSmith Hub](https://smith.langchain.com/hub) so you can see the updates and roll back prompts if desired.

![Demo](./img/demo.gif)

## Getting Started

To run the application locally, follow these steps:

Clone the repository:

```bash
git clone https://github.com/langchain-ai/tweet-critic.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Set up the necessary environment variables:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-api-key>
LANGCHAIN_HUB_API_KEY=<your-hub-api-key>
ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your web browser and navigate to the provided URL to access the application.

## Usage

Enter your initial tweet in the text box provided.
The application will generate a revised version of your tweet. Review the suggestion and provide feedback:

If you like the suggestion, click the thumbs up (👍) button. The modified tweet will be saved as an approved example.
If you dislike the suggestion, click the thumbs down (👎) button or respond to the chat bot with the requested changes.

If the suggestion is close but needs modification, edit the tweet directly in the text box before clicking the thumbs up button.

**Note:** Once you click either 👍 or 👎, the conversation will be finished, so the bot won't be permitted to respond further.

Continue the conversation by entering additional tweets or revisions. The application will learn from your feedback and adapt its suggestions accordingly.

To reset the session and start a new conversation, click the "Reset" button or refresh your browser.

## Customization

To modify the dataset used for few-shot examples, update the DATASET_NAME variable in the code.
To change the number of few-shot examples used, adjust the NUM_FEWSHOTS variable.

To use a different model for the main tweet critic or the optimizer, update the chat_llm and optimizer_llm variables respectively.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

#### License

This project is licensed under the MIT License.
