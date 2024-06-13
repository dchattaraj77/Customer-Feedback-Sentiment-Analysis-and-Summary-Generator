#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Customer Feedback Sentiment Analysis and Summary Generator


# In[ ]:


import openai
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# Load the .env file
_ = load_dotenv(find_dotenv())

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


# In[ ]:


# Helper function to get the completion from the model
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


# In[ ]:


# Function to analyze a single review
def analyze_review(review):
    summary_prompt = f"""
    What is the summary of the following product review in 2 short sentences?
    Review text: '{review}'
    """
    
    sentiment_prompt = f"""
    Determine the sentiment of the following product review for business and marketing insights. Respond with 'positive' or 'negative'.
    Review text: '{review}'
    """
    
    summary = get_completion(summary_prompt).strip()
    sentiment = get_completion(sentiment_prompt).strip().capitalize()  # Ensure proper capitalization
    
    return summary, sentiment


# In[ ]:


# Load the input CSV file
input_csv_path = 'customer_feedback.csv'
df = pd.read_csv(input_csv_path)


# In[ ]:


# Prepare lists to hold the summaries and sentiments
summaries = []
sentiments = []


# In[ ]:


# Iterate over each review in the CSV
for review in df['Review']:
    summary, sentiment = analyze_review(review)
    summaries.append(summary)
    sentiments.append(sentiment)


# In[ ]:


# Add the summaries and sentiments to the DataFrame
df['Summary'] = summaries
df['Sentiment'] = sentiments


# In[ ]:


# Save the updated DataFrame to a new CSV file
output_csv_path = 'customer_feedback_processed.csv'
df.to_csv(output_csv_path, index=False)


# In[ ]:


# Display the DataFrame to verify
df.head()

