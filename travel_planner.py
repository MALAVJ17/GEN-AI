import os
import requests
import streamlit as st
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import ChatLiteLLM
from dotenv import load_dotenv

load_dotenv()

# Set API keys
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up the Groq LLM via LiteLLM
llm = ChatLiteLLM(
    model="groq/llama3-70b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

# Weather Tool
@tool
def get_weather(location: str) -> Dict[str, Any]:
    """Get current weather for a location."""
    response = requests.get(
        f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}&aqi=no"
    )
    if response.status_code == 200:
        data = response.json()
        return {
            "location": f"{data['location']['name']}, {data['location']['country']}",
            "temperature_celsius": data["current"]["temp_c"],
            "condition": data["current"]["condition"]["text"],
            "feels_like": data["current"]["feelslike_c"]
        }
    else:
        return {"error": "Could not fetch weather data"}

# Search Tool (Tavily)
search_tool = TavilySearchResults(k=5)

# List of tools
tools = [get_weather, search_tool]

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel assistant. Use tools to provide accurate information."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create Agent and Executor
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI
st.set_page_config(page_title="Intelligent Travel Assistant AI", page_icon="🧳")

st.title("🧭 Travel Assistant AI")
location = st.text_input("Enter a destination city:")

if st.button("Show Weather & Attractions") and location:
    user_input = f"Show me the current weather and top attractions in {location}"
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_input})
        st.markdown("### 📝 Result")
        st.write(response["output"])
