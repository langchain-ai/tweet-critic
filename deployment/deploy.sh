#!/bin/bash

FUNCTION_NAME="tweet-critic-slackbot"
RUNTIME="python311"
ENTRY_POINT="handle_message"

# Set the necessary environment variables
ENV_VARS="LANGCHAIN_API_KEY=$LANGCHAIN_API_KEY,LANGCHAIN_HUB_API_KEY=$LANGCHAIN_HUB_API_KEY,ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY,LANGCHAIN_TRACING_V2=$LANGCHAIN_TRACING_V2,SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN,SLACK_APP_TOKEN=$SLACK_APP_TOKEN"

# Set your Google Cloud project ID
PROJECT_ID="langchain-test"

# Deploy the function
gcloud functions deploy $FUNCTION_NAME \
    --project $PROJECT_ID \
    --runtime $RUNTIME \
    --entry-point $ENTRY_POINT \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars $ENV_VARS