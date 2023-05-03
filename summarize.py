from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter

# CONSTANTS
FILENAME = "sem4_tris-fouilles"
FILEEXTENSION = ".pdf"

# UTILITY FUNCTIONS


def append_to_file(file_name, string):
    with open(file_name, "a") as file:
        file.write(string)


def read_file(file_name):
    with open(file_name, "r") as file:
        return file.read()


def file_exists(file_name):
    try:
        with open(file_name, "r") as file:
            return True
    except:
        return False

# ------------------------------
# SCRIPT
# ------------------------------


# TEMPLATES
CONCISE_SUMMARY_TEMPLATE = """
Write a concise summary of the following:


"{text}"


CONCISE SUMMARY:"""

REFINE_SUMMARY_TEMPLATE = """
Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {summary}
We have the opportunity to refine the existing summary(only if needed) with some more context below.
------------
{context}
------------
Given the above context, refine the existing summary
If the context isn't useful, return the existing summary.

FINAL SUMMARY:
"""

# SETUP THE LLM MODEL WE WILL USE
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=2000)

# SETUP A CONCISE SUMMARY CHAT CHAIN
concise_summary_human_prompt = HumanMessagePromptTemplate.from_template(
    CONCISE_SUMMARY_TEMPLATE)
concise_summary_chat_prompt = ChatPromptTemplate.from_messages(
    [concise_summary_human_prompt])
concise_summary_chain = LLMChain(
    llm=chat, prompt=concise_summary_chat_prompt, verbose=True)

# SETUP A REFINE SUMMARY CHAT CHAIN
refine_summary_human_prompt = HumanMessagePromptTemplate.from_template(
    REFINE_SUMMARY_TEMPLATE)
refine_summary_chat_prompt = ChatPromptTemplate.from_messages(
    [refine_summary_human_prompt])
refine_summary_chain = LLMChain(
    llm=chat, prompt=refine_summary_chat_prompt, verbose=True)

# SETUP THE DOCUMENT WE WILL SUMMARIZE
filename = FILENAME + FILEEXTENSION
loader = PyPDFLoader(filename)
document = loader.load_and_split()

# SETUP A TEXT SPLITTER TO SPLIT TEXT INTO CHUNKS WHEN NEEDED
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator='\n\n', chunk_size=750, chunk_overlap=150)

# ------------------------------
# PROCESSING STEP 1: LAYERED_MAP_REDUCE
# ------------------------------

# OUR OUTPUT SHALL BE STORED IN A LIST OF LAYERS
# EACH LAYER WILL BE A STRING (FOR NOW)
# LAYERS REPRESENT LEVELS OF ABSTRACTION WITH 0 BEING THE LEVEL OF ABSTRACTION RIGHT BELOW THE DOCUMENT
output_layers = []

# INITIALIZE SOME VARIABLES
split_document = document
layer_number = 0

while True:
    # CHECK IF THE LAYER SUMMARY TEMP FILE EXISTS
    if file_exists(f"{FILENAME}_layer_{layer_number}.txt"):
        output_layers.append(read_file(f"{FILENAME}_layer_{layer_number}.txt"))
    else:
        # IF IT DOESN'T EXIST, WE NEED TO GENERATE IT
        new_layer = ""
        for section in split_document:
            if layer_number == 0:
                # THE FIRST LAYER IS THE DOCUMENT ITSELF AND THUS HAS A DIFFERENT OBJECT TYPE
                input_text = section.page_content
            else:
                # ALL OTHER LAYERS ARE JUST STRINGS
                input_text = section

            # FORMAT THE PROMPT TO FEED THE LLM TO GENERATE A CONCISE SUMMARY FOR THE SECTION
            concise_summary_formated_chat_prompt = concise_summary_chat_prompt.format_prompt(
                text=input_text).to_messages()

            # RUN THE CHAIN TO GENERATE SAID SUMMARY
            section_summary = concise_summary_chain.run(
                text=input_text) + '\n\n'

            # APPEND THE SUMMARY TO THE CURRENT LAYER
            new_layer += section_summary

            # APPEND THE SUMMARY TO THE TEMP TEXT FILE
            append_to_file(
                f"{FILENAME}_layer_{layer_number}.txt", section_summary)

        # APPEND THE LAYER TO THE OUTPUT LAYERS LIST
        output_layers.append(new_layer)

    # SPLIT THE RESULTING LAYER INTO CHUNKS
    split_document = text_splitter.split_text(output_layers[-1])

    # CHECK IF THE RESULTING SPLIT DOCUMENT IS SHORT ENOUGH
    if len(split_document) < 5:
        print("Split document is short enough, proceeding to next step...")
        # IF IT IS, WE ARE DONE AND READY TO PROCEED TO THE NEXT STEP
        break
    else:
        # IF NOT INCREMENT THE LAYER NUMBER
        layer_number += 1

# ------------------------------
# PROCESSING STEP 2: REFINE_SUMMARY
# ------------------------------

# WE DEFINE SOME USEFUL VARIABLES
refine_input_array = split_document
refine_history = []
refine_step = 0

while True:
    # CHECK IF THE CURRENT STEP SUMMARY TEMP FILE EXISTS
    if file_exists(f"{FILENAME}_refine_step_{refine_step}.txt"):
        existing_summary = read_file(
            f"{FILENAME}_refine_step_{refine_step}.txt")
    else:
        # IF IT DOESN'T EXIST, WE NEED TO GENERATE IT
        summary = ""
        # IF IT IS THE FIRST REFINE STEP, WE USE THE CONCISE SUMMARY CHAIN
        if refine_step == 0:
            concise_summary_formated_chat_prompt = concise_summary_chat_prompt.format_prompt(
                text=refine_input_array[refine_step]).to_messages()

            # RUN THE CHAIN TO GENERATE SAID SUMMARY
            summary = concise_summary_chain.run(
                text=refine_input_array[refine_step])

            # APPEND THE SUMMARY TO THE REFINE HISTORY
            refine_history.append(summary)

            # APPEND THE SUMMARY TO THE TEMP TEXT FILE
            append_to_file(
                f"{FILENAME}_refine_step_{refine_step}.txt", summary)
        else:
            # IF IT IS NOT THE FIRST REFINE STEP, WE USE THE REFINED SUMMARY CHAIN
            refine_summary_formated_chat_prompt = refine_summary_chat_prompt.format_prompt(
                summary=refine_history[-1], context=refine_input_array[refine_step]).to_messages()

            # RUN THE CHAIN TO GENERATE SAID SUMMARY
            summary = refine_summary_chain.run(
                summary=refine_history[-1], context=refine_input_array[refine_step])

            # APPEND THE SUMMARY TO THE REFINE HISTORY
            refine_history.append(summary)

            # APPEND THE SUMMARY TO THE TEMP TEXT FILE
            append_to_file(
                f"{FILENAME}_refine_step_{refine_step}.txt", summary)

    # IF IT WAS THE LAST STEP, WE ARE DONE
    if refine_step == len(refine_input_array) - 1:
        print("Refine step is the last step, we are done!")
        break
    else:
        # OTHERWISE, WE INCREMENT THE STEP NUMBER
        refine_step += 1
