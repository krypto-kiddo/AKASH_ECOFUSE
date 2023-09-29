# ADD YOUR API KEYS, MODEL CONFIGURATION VALUES AND CUSTOM PROMPTS BEFORE RUNNING THE APPLICATION
# Programmed by @Chaitanyarai899 , @krypto-kiddo and @boldpanther

import requests
import streamlit as st

import os
import subprocess
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

st.write("# 🤖: Hi! I am Akash Ecofuse Autodeployer.")
st.write("### I just need the following info to automatically deploy your project 🚀")
st.write("---------------------------------------------------------------------------")

_left, _right = st.columns(2)

with _left:
  user = st.text_input("Github Username", placeholder="JohnDoe69")
with _right:
  repo = st.text_input("Github Repository Name",placeholder="sample-project")
dockerfile_instructions = st.text_input("This is optional: Some Additional requirements for you dockerfile like Port number to Expose, even techstack used, commands to start etc. Basically input here for a very precise dockerfile with all your requirments.","-")
with _left:
  dockerhub_username = st.text_input("Docker Hub Username", placeholder="JohnDoe420")
with _right:
  dockerhub_password = st.text_input("Docker Hub Password", type="password", placeholder="*****")
with _left:
  dockerhub_repo_name = st.text_input("Docker Hub Repository Name", placeholder="current-repo")
user_requirment = st.text_input("Additional Requirements for your sdl file, like some env variables, your budget, the amount of ram you need and any other specifition", placeholder="Eg: My budget is 2AKT, Need 2GB ram, Windows OS, etc.")

if st.button("Genrate SDL File 🚀"):
    if user and repo:
        repo_url = f"https://github.com/{user}/{repo}.git"

        with st.spinner(' ### 🤖: I am Cloning your repository ..... hmmm Looks interesting'):
            try:
                result = subprocess.run(["git", "clone", repo_url], check=True, text=True, capture_output=True)
                st.success(f"Repository cloned successfully")
            except subprocess.CalledProcessError as e:
                st.success(f"🤖: I have already cloned your repo!")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.info("Enter the GitHub username and repository name.")

    repo_url = "https://api.github.com/repos/{owner}/{repo_name}/contents/{folder_path}"

    owner = user
    repo_name = repo

    print(owner, repo_name)

    folder_path = ""
    markdown_structure = ""

    def fetch_and_build_markdown_structure(owner, repo_name, folder_path, depth=0):
      global markdown_structure
      repo_url = "https://api.github.com/repos/{owner}/{repo_name}/contents/{folder_path}"
      url = repo_url.format(owner=owner, repo_name=repo_name,  folder_path=folder_path)
      token = "<GITHUB ACCESS TOKEN HERE>"
      '''headers = {
          "Authorization": f"token {token}",
          "User-Agent": "<GITHUB AGENT NAME HERE>"
      }'''
      auth = ("<GITHUB AGENT NAME HERE>",token)
      response = requests.get(url, auth=auth) # Implemented tokenized access

      if response.status_code == 200:
          data = response.json()
          if isinstance(data, list):
              for item in data:
                  indent = "  " * depth
                  if item["type"] == "dir":
                      folder_name = item["name"]
                      markdown_structure += f"{indent}- Folder: {folder_name}\n"
                      fetch_and_build_markdown_structure(owner, repo_name, item["path"], depth + 1)
                  elif item["type"] == "file":
                      file_name = item["name"]
                      # Exclude files with certain extensions
                      if not any(file_name.endswith(ext) for ext in [".png", ".gif", ".jpeg", ".jpg", ".mp3", ".mp4",".svg"]):
                          markdown_structure += f"{indent}- File: {file_name}\n"
          else:
              print("Invalid response data")
      else:
          print(f"Failed to fetch folder structure. Status code: {response.status_code}")

      return markdown_structure

    with st.spinner('Fetching and building markdown structure...'):
        fetch_and_build_markdown_structure(owner, repo_name, folder_path)

    Proceedtosdl = False

    if markdown_structure:
        st.write(" ### 🤖: Here is how the folder structure of your repo looks like -")
        st.code(markdown_structure, language='markdown')

    if markdown_structure:
        from langchain import PromptTemplate, OpenAI, LLMChain

        prompt_template = "< ENTER GENERATION PROMPT HERE >"

        llm = OpenAI(temperature="<ENTER CUSTOM TEMPERATURE VALUE AS AN INTEGER HERE>", openai_api_key="<OPEN API KEY HERE>")
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )

    with st.spinner('### 🤖: Hang tight I am Generating Dockerfiles and docker-compose file.....'):
      if markdown_structure:
        from langchain import PromptTemplate, OpenAI, LLMChain

        prompt_template = "<ENTER GENERATION PROMPT HERE> :"+ dockerfile_instructions

        llm = OpenAI(temperature=0, openai_api_key="<OPEN API KEY HERE>")
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )
        res = llm_chain(markdown_structure)
        res = res["text"]
        sdlprompt = res
        st.write(" ### 🤖: Here is the Dockerfile and docker-compose file you need!")
        sections = res.split("Dockerfile:")

        repo_path = repo_name

        for i, section in enumerate(sections[1:], start=1):
            if "docker-compose.yml" in section:
                section = section.split("docker-compose.yml")[0]
            st.markdown("------------------------------------------------------------------------------")
            st.code(section, language='markdown')

            # Save Dockerfile in the root folder of the cloned repository
            with open(os.path.join(repo_path, 'Dockerfile'), 'w') as file:
                file.write(section)

        if "docker-compose.yml" in res:
            docker_compose_section = res.split("docker-compose.yml")[-1]
            st.markdown("------------------------------------------------------------------------------")
            st.code(f"{docker_compose_section}", language='markdown')

            with open(os.path.join(repo_path, 'docker-compose.yml'), 'w') as file:
                file.write(docker_compose_section)


    if dockerhub_username and dockerhub_repo_name and dockerhub_password:

        image_name = f"{dockerhub_username}/{dockerhub_repo_name}:latest"

        if dockerhub_username and dockerhub_password and dockerhub_repo_name:
                # Login to Docker Hub 
                result = subprocess.run(["docker", "login", "-u", dockerhub_username, "-p", dockerhub_password], check=True, capture_output=True)

                # Build the Docker image
                with st.spinner('### 🤖: I am Building the Docker image...... installing the needed things to run your beautifully written code in the container...'):
                    try:
                      result = subprocess.run(["docker", "build", "-t", image_name, repo_path], check=True, capture_output=True)
                      st.success(f"### 🤖: Image {image_name} built successfully!")
                    except subprocess.CalledProcessError as e:
                        st.error(f"Failed to build the image. Error: {e.stderr.decode('utf-8')}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")

                # Push the Docker image to Docker Hub
                with st.spinner('### 🤖: I am now Pushing the Docker image to your Docker Hub Account 🚀.....'):
                    result = subprocess.run(["docker", "push", image_name], check=True, capture_output=True)
                    st.success(f"### 🤖: Image {image_name} pushed to Docker Hub successfully 🦾")
                    Proceedtosdl = True

    else:
        st.info("🤖: Please provide all the details and click Submit.")

    def create_sdl_file(user_requirment):
         finalsdlprompt =  sdlprompt + "<ENTER YOUR SDL GENERATION PROMPT HERE>"
         print(finalsdlprompt)
         embeddings = OpenAIEmbeddings(model="<ENTER TEXT MODEL>",openai_api_key="<OPENAI API KEY>")
         vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma")
         retriever = vectorstore.as_retriever()

         memory = ConversationBufferMemory(
             memory_key='chat_history',
             return_messages=True
         )

      # Setup LLM and QA chain
         llm = ChatOpenAI(model_name="<ENTER MODEL NAME HERE>", temperature="<ENTER TEMPERATURE AS AN INTEGER>", streaming=True, openai_api_key='<OPENAI API KEY>')
         system_template = """
         
         <ENTER A PRE-PROMPT HERE>

         Here is a SDL template for your reference:
         # Akash Network Service Description Language (SDL) File

# Service Metadata
version: '2.0'
services:
  my-service:
    image: [docker_image]
    expose:
      - port: [exposed_port]
        as: 80
        to:
          - global: true
profiles:
  compute:
    my-service:
      resources:
        cpu:
          units: [cpu_units]
        memory:
          size: [memory_size]
        storage:
          - size: [storage_size]
        gpu:
          units: [gpu_units]
          attributes:
            vendor:
              nvidia:
                - model: [gpu_model]
  placement:
    akash:
      pricing:
        my-service:
          denom: uakt
          amount: [akash_price]
deployment:
  my-service:
    akash:
      profile: my-service
      count: [instance_count]

         ----------------
         {context}
         ----------------
         <ENTER POST-PROMPT HERE TO FILTER OUT SDL FILE FROM OTHER EXTRA INFORMATION IN THE REPLY MESSAGE>
         """

      # Create the chat prompt templates
         messages = [
         SystemMessagePromptTemplate.from_template(system_template),
         HumanMessagePromptTemplate.from_template("{question}")
         ]
         qa_prompt = ChatPromptTemplate.from_messages(messages)

         qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever,
                                                           memory=memory,
                                                            verbose=True,
                                                            combine_docs_chain_kwargs={"prompt": qa_prompt})

         sdlfile = qa_chain(finalsdlprompt)

         return sdlfile

    if Proceedtosdl==True:
        if user_requirment != 'empty':
          with st.spinner("### 🤖: Now, I am writing your sdl file, hang tight.."):
            genratedsdlfile = create_sdl_file(user_requirment)
            display = genratedsdlfile['answer']
            print(genratedsdlfile)

            st.write("### 🤖: Here is your sdl file...It was an absolute pleasure working for you. Hope we meet again soon! 👉👈")
            st.code(display, language='markdown')

# END OF CODE
