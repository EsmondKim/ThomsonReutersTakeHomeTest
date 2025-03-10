import csv
import os
from bs4 import BeautifulSoup
import pandas as pd
import requests
import json
from tabulate import tabulate
from openai import OpenAI
import time

def extract_negative_treatments(slug):
    # While multiple standard headers are provided to the API, it is specifically the "Referer" key and value that makes the api call in this script work.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://casetext.com/",
        "Connection": "keep-alive"
    }

    url = f"https://casetext.com/api/search-api/doc/{slug}/html"
    max_retries=3 
    backoff_factor=2

    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            analyzed_case = slug
            soup = BeautifulSoup(response.text, 'html.parser')

            # This line is using list comprehension and beautiful soup to add the text of each <p> element to the list as one string.
            paragraphs = [p.text for p in soup.find_all('p')]
            
            # max_tokens = 128000 # Currently, the context limit in GPT-4o is 128000
            max_tokens = 10000 

            # " " is a delimiter, meaning Python will iterate through the list of separate strings, and combine them into a single string, with a space delimiting each paragraph.
            combined_text = " ".join(paragraphs)
            # This line will create a Python list of every individual word as one string, identifying words by white space.
            tokens = combined_text.split()
            # This line will join the individual words back into one long string, up the max_tokens limit we've set, by using a Python slice operation.
            paragraphs = " ".join(tokens[:max_tokens])
            # print("this is paragraphs, after all that work", paragraphs)

            if not paragraphs:
                print("Warning: Extracted text is empty. Check API response.")
                return None

            return paragraphs, analyzed_case
  
        elif response.status_code in [500, 502, 503, 504]:  # Retry-worthy errors
            print(f"Server error {response.status_code}. Retrying in {backoff_factor ** attempt} seconds...")
            time.sleep(backoff_factor ** attempt)
        else:
            print(f"Failed to fetch HTML. Status code: {response.status_code}")
            return None  # Return None instead of stopping execution
        print("Max retries reached. Exiting function.")
        return None

client = OpenAI(api_key="")

def query_chatgpt_negative_treatment(paragraphs):
    prompt = f"""
    Here is some text content from a legal opinion:

    {paragraphs}...

    Please analyze the document for negative legal treatment, specifically focusing on any cases where terms like "overrule" and "reject" are used, or similar expressions like "reversed," "vacated," or "set aside." If the document says, for example, "We overrule Alfree and reject the Doctrine as a defense in this case," this should be flagged as negative treatment of Alfree.

    Focus specifically on identifying **cases where the instant case reverses, vacates, distinguishes, or criticizes** the ruling, opinion, or legal reasoning of prior cases, even if these terms are not explicitly used in the document.

    **Negative treatment** refers to actions like:
    - **Reversing, vacating, overruling**, or **distinguishing** prior legal decisions.
    - Criticizing or rejecting the reasoning or application of another case’s legal principles.

    Ensure that you identify **only** instances where the immediate case explicitly changes or rejects the holding, ruling, or reasoning of another case.

    Your response should be in JSON format with the following structure:
    {{
      "negative_treatment": true or false,
      "nature_of_treatment": A concise summary of the negative treatment (e.g., 'We reject the holding of Devereaux' or 'we overrule Alfree and reject the Doctrine as a defense in this case'),
      "excerpts": A single string of full-sentence excerpts indicating negative treatment, separated by newlines. If none found, return 'Not applicable',
      "rationale": A detailed explanation of why negative treatment was or wasn’t found, referring to specific legal reasoning. If no negative treatment is found, return 'Not applicable'
    }}
    Ensure that your response is a **valid JSON object** and contains **no additional text or commentary**.
    """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
           {
               "role": "system",
                "content": """You are an AI specializing in legal case law analysis. 
                Your task is to determine whether a given court's opinion applies negative treatment to caselaw that is cited within the court's opinion. 
                Negative legal treatment can include terms such as overruled, overturned, vacated, reversed, superseded, and similar expressions. 
                Here is an example of some case names and case law citations:
                    Ferguson v. Harrison, 34 S.C. 169; 13 S.E., 332
                    Devereux v. McCrady, 49 S.C. 423; 27 S.E., 467
                    Williams v. Newton, 84 S.C. 100; 65 S.E., 959
                    Luna v. Clayton, Tenn.Supr., 655 S.W.2d 893, 896-97 (1983)
                    Daubert v. Merrell Dow Pharmaceuticals, Inc., ___ U.S. ___, 113 S.Ct. 2786, 125 L.Ed.2d 469 (1993)
                    Shearer v. Shearer, Oh.Supr., 18 Ohio St.3d 94, 97, 18 OBR 129, 131, 480 N.E.2d 388, 393 (1985)
                Often, negative treatment of one case against a prior case is indicated by the use of terms such as overruled, overturned, vacated, reversed, superseded, and similar expressions close to case names and case law citations.
                    
                Follow these rules:
                - Only analyze the provided legal opinion's text; do not infer beyond the given content.  
                - Always return output as a structured JSON object.  
                - Be objective and precise in identifying legal treatment.  
                - Before marking a case as receiving negative treatment, verify:
                    - That the legal opinion is explicitly negating, reversing, or distinguishing a prior case.
                    - That negation words (e.g., "overruled") are **not used metaphorically** or in a different legal context.
                    - That the reference is to **precedent cases** and not dicta or hypothetical scenarios.
                - If multiple cases are negatively treated in the opinion, list each case separately on newlines with its corresponding nature of treatment and excerpts. 
                - If uncertainty exists, err on the side of marking "negative_treatment": false.
                - If no negative treatment is found, explicitly return 'Not applicable' for the relevant fields.

                Your response must be a **valid JSON object** and contain **no additional explanations or commentary**."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    response_text = completion.choices[0].message.content

    if response_text.startswith("```json") and response_text.endswith("```"):
        # This Python Slice takes off the ChatCompletionMessage prefix of: ```json and the trailing ``` because that syntax creates an error with json.loads() method.
        response_text = response_text[7:-3].strip()
    try:
        response_json = json.loads(response_text)
        boolean_response = response_json["negative_treatment"]
        nature_of_treatment_response = response_json["nature_of_treatment"]
        caselaw_excerpts = response_json["excerpts"]
        rationale_response = response_json["rationale"]
    except json.JSONDecodeError:
        # Logs for json decoding errors and log of errored response_text
        print("Error: Could not parse JSON response. Raw response was:")
        print(response_text)
        boolean_response, nature_of_treatment_response, caselaw_excerpts, rationale_response = "False", "Not applicable", "Not applicable", "No rationale provided"

    return boolean_response, nature_of_treatment_response, caselaw_excerpts, rationale_response

# paragraphs, analyzed_case = extract_negative_treatments(slug='little')
# paragraphs, analyzed_case = extract_negative_treatments(slug='john-v-state-7')
# paragraphs, analyzed_case = extract_negative_treatments(slug='beattie-v-beattie')
# paragraphs, analyzed_case = extract_negative_treatments(slug='travelers-indem-co-v-lake')
# paragraphs, analyzed_case = extract_negative_treatments(slug='tilden-v-state')
paragraphs, analyzed_case = extract_negative_treatments(slug='in-re-lee-342013')

if paragraphs:
    boolean_response, nature_of_treatment_response, caselaw_excerpts, rationale_response = query_chatgpt_negative_treatment(paragraphs)

    table_data = [
        (analyzed_case, boolean_response, nature_of_treatment_response, caselaw_excerpts, rationale_response)
    ]

    table_df = pd.DataFrame(table_data, columns=['Treated Case', 'Negative Treatment Present', 'Nature of Treatment', 'Text of Negative Treatment', 'LLM Explanation'])

    # Log to verify creation of Pandas DataFrame
    print("\nDataFrame Created: ", tabulate(table_df, headers='keys', tablefmt='pretty'))

    file_path = rf"C:\Users\ekele\Desktop\Coding\ThomsonReutersPromptEngineerTakeHomeTest\exported_csvs\{analyzed_case}.csv"

    counter = 1
    original_file_path = file_path

    while os.path.exists(file_path):
        file_path = rf"C:\Users\ekele\Desktop\Coding\ThomsonReutersPromptEngineerTakeHomeTest\exported_csvs\{analyzed_case}({counter}).csv"
        counter += 1

    table_df.to_csv(file_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

    # Log to confirm successful creation and export of CSV.
    print(f"CSV successfully saved at: {file_path}")