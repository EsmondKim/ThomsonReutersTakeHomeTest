from bs4 import BeautifulSoup
import pandas as pd
import requests
import json
from tabulate import tabulate
from openai import OpenAI

def extract_negative_treatments(slug):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://casetext.com/",
        "Connection": "keep-alive"
    }

    url = f"https://casetext.com/api/search-api/doc/{slug}/html"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        treated_case = slug
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.text for p in soup.find_all('p')]
        
        return paragraphs, treated_case
    else:
        # Log to catch errors and provide API status code.
        print(f"Failed to fetch HTML. Status code: {response.status_code}")
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
                - If no negative treatment is found, explicitly return 'Not applicable' for the relevant fields.

                Your response must be a **valid JSON object** and contain **no additional explanations or commentary**."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
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

paragraphs, treated_case = extract_negative_treatments(slug='little')
# paragraphs, treated_case = extract_negative_treatments(slug='john-v-state-7')
# paragraphs, treated_case = extract_negative_treatments(slug='beattie-v-beattie')
# paragraphs, treated_case = extract_negative_treatments(slug='travelers-indem-co-v-lake')
# paragraphs, treated_case = extract_negative_treatments(slug='tilden-v-state')
# paragraphs, treated_case = extract_negative_treatments(slug='in-re-lee-342013')

if paragraphs:
    boolean_response, nature_of_treatment_response, caselaw_excerpts, rationale_response = query_chatgpt_negative_treatment(paragraphs)

    table_data = [
        (treated_case, boolean_response, nature_of_treatment_response, caselaw_excerpts, rationale_response)
    ]

    table_df = pd.DataFrame(table_data, columns=['Treated Case', 'Negative Treatment Present', 'Nature of Treatment', 'Text of Negative Treatment', 'LLM Explanation'])

    # Log to verify creation of Pandas DataFrame
    print("\nDataFrame Created: ", tabulate(table_df, headers='keys', tablefmt='pretty'))

    file_path = r"C:\Users\ekele\Desktop\ThomsonReutersPromptEngineerPosition\TakeHomeTestCodeFiles\exported_csv_files\output.csv"

    table_df.to_csv(file_path, index=False, encoding='utf-8')
    # Log to confirm successful creation and export of CSV.
    print(f"CSV successfully saved at: {file_path}")
