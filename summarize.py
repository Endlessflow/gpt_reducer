from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from templates import CONCISE_SUMMARY_TEMPLATE, REFINE_SUMMARY_TEMPLATE
from utils import append_to_file, file_exists, read_file

# CONSTANTS
FILENAME = "Cedille_transcript"
FILEEXTENSION = ".txt"

# ------------------------------
# SCRIPT
# ------------------------------

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

# SETUP A TEXT SPLITTER TO SPLIT TEXT INTO CHUNKS WHEN NEEDED
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator='\n\n', chunk_size=750, chunk_overlap=150)

# SOME USEFUL VARIABLES
filename = FILENAME + FILEEXTENSION

# ------------------------------
# PROCESSING STEP 1: LAYERED_MAP_REDUCE
# ------------------------------

# OUR OUTPUT SHALL BE STORED IN A LIST OF LAYERS
# EACH LAYER WILL BE A STRING (FOR NOW)
# LAYERS REPRESENT LEVELS OF ABSTRACTION WITH 0 BEING THE ORIGINAL TEXT
output_layers = []

# INITIALIZE SOME VARIABLES
document = []
layer_number = 0

while True:
    # CHECK IF THE LAYER SUMMARY TEMP FILE EXISTS
    if file_exists(f"{FILENAME}_layer_{layer_number}.txt"):
        output_layers.append(read_file(f"{FILENAME}_layer_{layer_number}.txt"))
    else:
        # IF IT DOESN'T EXIST, WE NEED TO GENERATE THE NEW LAYER
        new_layer = ""

        # IF THE LAYER NUMBER IS 0, WE FIRST NEED TO LOAD THE DOCUMENT
        # OTHERWISE WE ALREADY HAVE THE DOCUMENT LOADED FROM THE PREVIOUS LAYER
        if layer_number == 0:
                if FILEEXTENSION == ".pdf":
                    # IF THE FILE IS A PDF, WE NEED TO LOAD IT WITH THE PDF LOADER
                    loader = PyPDFLoader(filename)
                    document = loader.load_and_split()

                elif (FILEEXTENSION == ".txt"):
                    single_line_break_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(separator='.', chunk_size=750, chunk_overlap=0)
                    text = read_file(filename)
                    document = single_line_break_text_splitter.split_text(text)


        for section in document:
            
            if FILEEXTENSION == ".pdf":
                input_text = section.page_content
            else:
                # ALL OTHER LAYERS ARE JUST STRINGS
                input_text = section

            if layer_number == 0:
                # IF IT IS THE FIRST LAYER, WE DON'T NEED TO GENERATE A SUMMARY
                output = input_text + '\n\n'
            else:
                # FORMAT THE PROMPT TO FEED THE LLM TO GENERATE A CONCISE SUMMARY FOR THE SECTION
                concise_summary_formated_chat_prompt = concise_summary_chat_prompt.format_prompt(
                    text=input_text).to_messages()

                # RUN THE CHAIN TO GENERATE SAID SUMMARY
                output = concise_summary_chain.run(
                    text=input_text) + '\n\n'
                
                print(output)


            # APPEND THE SUMMARY TO THE CURRENT LAYER
            new_layer += output

            # APPEND THE SUMMARY TO THE TEMP TEXT FILE
            append_to_file(
                f"{FILENAME}_layer_{layer_number}.txt", output)

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
