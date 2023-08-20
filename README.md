# openai_chatbot_cms_
# Question-and-answer-using-document

## Project description
The application is created to leverage on openai and LLMs such as chatgpt-4 to accept user input and provide intelligent responses based on the content  given to it.
Our choice for openai and LLMs is based on current evolution in AI to process data in human like ways, give reasonable and meaningful responses that can help companies, organizations and individuals automate their processes.
The challenges with this technology is the billing based on token, and even if you can afford the expenses the token limitations comes in. As you be aware the most recent chatgpt, thats gpt-4 can only accept upto 50 pages. This gives a ceiling on what you can send to LLMs for processing. And therefore opens awindow to utilise  vectordatabase such chromadb, to break big documents into smaller chunks  and be stored as vectors of related information that can be called based on user input. Through matching.
For this project, we will not dive much into vectordatabase, since we will only be giving a handful of data into the code to be processed. But in our coming projects we will use larger documents that will need us to break them into chunk hence o using vectorstores.

## Pre requisites
- python 3.9+
- vs Code
- Visual Studio Build Tools-latest.
  Download and install using the link below.

  ```
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
  ```

## Project setup

clone the repo

```
  git clone https://github.com/ovesorg/openai_chatbot_cmss_.git
```
 install the requirements for the project by running the following on git terminal

```
 pip install -r requirements.txt
```
 


## Run the project
remember to replace 'openai_api-key' with your own relevant and legit keys.
You can change the content by copying any text of less than one page and insert in the content and query against it.


```

  python chatbot.py
```
