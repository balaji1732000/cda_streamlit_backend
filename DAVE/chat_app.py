import base64
import os
import time
import streamlit as st
from openai import AzureOpenAI
from openai.types.beta.assistant_stream_event import (
    ThreadRunStepCreated,
    ThreadRunStepDelta,
    ThreadRunStepCompleted,
    ThreadMessageCreated,
    ThreadMessageDelta,
)
from openai.types.beta.threads.text_delta_block import TextDeltaBlock
from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
from openai.types.beta.threads.runs.code_interpreter_tool_call import (
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs,
)

# Set page config
st.set_page_config(page_title="DAVE", layout="wide")

# Create Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint="https://dwspoc.openai.azure.com/",
    api_key="6e27218cd3eb4dd0b0d94277679b79e4",
    api_version="2024-02-15-preview",
)

ASSISTANT_ID = "asst_GHCAbg0ZKHVd8lFcAJG4kq61"

# Apply custom CSS
st.markdown(
    """  
    <style>  
        .MainMenu, footer, header, .css-1oe5cao, .css-1v3fvcr, .css-1n543e5 {visibility: hidden;}  
        .css-18e3th9 { padding: 0 1rem; }  
        .css-1lcbmhc, .css-1d391kg { padding: 0 1rem; max-width: 100% !important; }  
        .selectbox-label { color: black; font-weight: bold; }  
        .main { background-color: #304666; }  
        [data-testid="stStatusWidget"] { visibility: hidden; }  
        .stTabs [data-baseweb="tab"]:hover { background-color: #f0f0f0; color: #000; }  
        .stTabs [data-baseweb="tab"][aria-selected="true"]{ background-color: #fff; color: #000; border-bottom: 2px solid #ff4b4b; }  
        .css-1d391kg { font-family: 'Arial', sans-serif; font-size: 1rem; }  
        .css-1hb8ztp { font-size: 1rem; color: #333; }  
        .css-10trblm { padding: 0.5rem; }  
        .title { font-size: 3rem; font-weight: bold; color: #002B50; text-align: center; font-family: 'Arial', sans-serif; background-color: #F0F8FF; padding: 1rem; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; }  
        .black-subheader h3, .black-subheader h4 { color: white; text-align: center; }  
        div[data-testid="stSelectbox"] p { color: white; }  
        .block-container { background-color: #304666 !important; }  
        .css-1aumxhk { background-color: #304666 !important; }  
        .stTextInput input { background-color: #304666 !important;} 

        .stBottomBlockContainer { background-color: #304666 }  
        div[data-testid="stMarkdownContainer"] p #text li ul ol { color: white }

    </style>  
    """,
    unsafe_allow_html=True,
)

# Initialise session state
for session_state_var in ["file_uploaded"]:
    if session_state_var not in st.session_state:
        st.session_state[session_state_var] = False


# Moderation check
def moderation_endpoint(text) -> bool:
    response = client.moderations.create(input=text)
    return response.results[0].flagged


# UI
file_upload_box = st.empty()
upload_btn = st.empty()

# File Upload
if not st.session_state["file_uploaded"]:
    st.session_state["files"] = file_upload_box.file_uploader(
        "Please upload your dataset(s)", accept_multiple_files=True, type=["csv"]
    )
    if upload_btn.button("Upload"):
        st.session_state["file_id"] = []
        for file in st.session_state["files"]:
            try:
                oai_file = client.files.create(file=file, purpose="assistants")
                st.session_state["file_id"].append(oai_file.id)
                print(f"Uploaded new file: {oai_file.id}")
            except Exception as e:
                print(f"Failed to upload file: {str(e)}")
        st.toast("File(s) uploaded successfully", icon="🚀")
        st.session_state["file_uploaded"] = True
        file_upload_box.empty()
        upload_btn.empty()
        st.rerun()

if st.session_state["file_uploaded"]:
    if "thread_id" not in st.session_state:
        try:
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": "Can you provide a brief understanding about my dataset",
                        "file_ids": [file_id for file_id in st.session_state.file_id],
                    }
                ]
            )
            st.session_state.thread_id = thread.id
            print(st.session_state.thread_id)
        except Exception as e:
            print(f"Failed to create thread: {str(e)}")

    client.beta.threads.update(thread_id=st.session_state.thread_id)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for item in message["items"]:
                if item["type"] == "text":
                    st.markdown(item["content"])
                elif item["type"] == "image":
                    for image in item["content"]:
                        st.html(image)
                elif item["type"] == "code_output":
                    with st.status("Results", state="complete"):
                        st.code(item["content"])

    if prompt := st.chat_input("Ask me a question about your dataset"):
        st.session_state.messages.append(
            {
                "role": "user",
                "items": [
                    {
                        "type": "text",
                        "content": prompt,
                    }
                ],
            }
        )
        try:
            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt,
            )
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message(
                "assistant",
                # avatar=st.image(
                #     "F:\Documents backup\AI Projects\CDA_V2\CDA_V2\CDA-master 2\backend\DAVE\images\bot.png"
                # ),
            ):
                stream = client.beta.threads.runs.create(
                    thread_id=st.session_state.thread_id,
                    assistant_id=ASSISTANT_ID,
                    tool_choice={"type": "code_interpreter"},
                    stream=True,
                )
                assistant_output = []
                for event in stream:
                    print(event)
                    if isinstance(event, ThreadRunStepCreated):
                        if event.data.step_details.type == "tool_calls":
                            assistant_output.append(
                                {"type": "code_input", "content": ""}
                            )
                            code_input_expander = st.status(
                                "Writing code ⏳ ...", expanded=True
                            )
                            code_input_block = code_input_expander.empty()
                    if isinstance(event, ThreadRunStepDelta):
                        if (
                            event.data.delta.step_details.tool_calls[0].code_interpreter
                            is not None
                        ):
                            code_interpretor = event.data.delta.step_details.tool_calls[
                                0
                            ].code_interpreter
                            code_input_delta = code_interpretor.input
                            if (code_input_delta is not None) and (
                                code_input_delta != ""
                            ):
                                assistant_output[-1]["content"] += code_input_delta
                                code_input_block.empty()
                                code_input_block.code(assistant_output[-1]["content"])
                    elif isinstance(event, ThreadRunStepCompleted):
                        if isinstance(event.data.step_details, ToolCallsStepDetails):
                            code_interpretor = event.data.step_details.tool_calls[
                                0
                            ].code_interpreter
                            if code_interpretor.outputs:
                                code_interpretor_outputs = code_interpretor.outputs[0]
                                code_input_expander.update(
                                    label="Code", state="complete", expanded=False
                                )
                                if isinstance(
                                    code_interpretor_outputs, CodeInterpreterOutputImage
                                ):
                                    image_html_list = []
                                    for output in code_interpretor.outputs:
                                        image_file_id = output.image.file_id
                                        image_data = client.files.content(image_file_id)
                                        image_data_bytes = image_data.read()
                                        with open(
                                            f"images/{image_file_id}.png", "wb"
                                        ) as file:
                                            file.write(image_data_bytes)
                                        file_ = open(
                                            f"images/{image_file_id}.png", "rb"
                                        )
                                        contents = file_.read()
                                        data_url = base64.b64encode(contents).decode(
                                            "utf-8"
                                        )
                                        file_.close()
                                        image_html = f'<p align="center"><img src="data:image/png;base64,{data_url}" width=600></p>'
                                        st.html(image_html)
                                        image_html_list.append(image_html)
                                    assistant_output.append(
                                        {"type": "image", "content": image_html_list}
                                    )
                                elif isinstance(
                                    code_interpretor_outputs, CodeInterpreterOutputLogs
                                ):
                                    assistant_output.append(
                                        {"type": "code_output", "content": ""}
                                    )
                                    code_output = code_interpretor.outputs[0].logs
                                    with st.status("Results", state="complete"):
                                        assistant_output[-1]["content"] = code_output
                            else:
                                print("No outputs generated by the code interpreter.")
                    elif isinstance(event, ThreadMessageCreated):
                        assistant_output.append({"type": "text", "content": ""})
                        assistant_text_box = st.empty()
                    elif isinstance(event, ThreadMessageDelta):
                        if isinstance(event.data.delta.content[0], TextDeltaBlock):
                            assistant_text_box.empty()
                            assistant_output[-1]["content"] += event.data.delta.content[
                                0
                            ].text.value
                            assistant_text_box.markdown(assistant_output[-1]["content"])
                st.session_state.messages.append(
                    {"role": "assistant", "items": assistant_output}
                )
        except Exception as e:
            # st.error(f"Error during processing: {str(e)}")
            print(f"Error during processing: {str(e)}")
