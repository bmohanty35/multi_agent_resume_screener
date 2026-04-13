from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
import os

import warnings
warnings.filterwarnings("ignore")

load_dotenv()
# SET YOUR API KEY HERE
os.environ["GROQ_API_KEY"] = os.getenv("groq_key")

MODEL_NAME = "llama-3.1-8b-instant"
# ========================
# MODEL
# ========================
llm = ChatGroq(model=MODEL_NAME)

# ========================
# TOOLS
# ========================
search = DuckDuckGoSearchResults(max_results=2)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ========================
# AGENTS
# ========================

skills_agent = create_agent(
    model=llm,
    tools=[wiki],
    system_prompt="""You are a helpful assistant with access to tools.
    Use tools via function calls to answer questions.

    Be concise. Do not loop. Return final answer quickly."""
)

experience_agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt=""" You are a helpful assistant with access to tools.
    Use tools via function calls to answer questions.

    Be concise. Do not loop. Return final answer quickly."""
)

salary_agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="""You are a helpful assistant with access to tools.
    Use tools via function calls to answer questions.

    Be concise. Do not loop. Return final answer quickly."""
)

# ========================
# TOOL FUNCTIONS
# ========================

@tool
def call_skills_matcher(resume: str, job_desc: str) -> str:
    """Skills matching analysis"""
    try:
        response = skills_agent.invoke({
            "messages": [HumanMessage(content=f"""
            Resume skills: {resume[:200]}
            Job requires: {job_desc[:200]}
            Return: MATCH: XX% | MISSING: [list] | RATING: X/10
            """)]
        })
        return response["messages"][-1].content
    except Exception as e:
        return "Unable to analyze at the moment. Please try again."


@tool
def call_experience_evaluator(resume_exp: str, job_level: str) -> str:
    """Experience evaluation"""
    try:
        response = experience_agent.invoke({
            "messages": [HumanMessage(content=f"""
            Experience: {resume_exp[:200]}
            Job level: {job_level}
            Return: FIT: XX% | STRENGTHS: ... | WEAKNESS: ...
            """)]
        })
        return response["messages"][-1].content
    except Exception as e:
        return "Unable to analyze at the moment. Please try again."

@tool
def call_salary_researcher(role: str, location: str, years: int) -> str:
    """Salary market research"""
    try:
        response = salary_agent.invoke({
            "messages": [HumanMessage(content=f"""
            Role: {role}, Location: {location}, Years: {years}
            Return: RANGE: ₹X-YL | PERCENTILE: Xth
            """)]
        })
    except Exception as e:
        return "Unable to analyze at the moment. Please try again."


# ========================
# SUPERVISOR
# ========================

supervisor_agent = create_agent(
    model=llm,
    tools=[
        call_skills_matcher,
        call_experience_evaluator,
        call_salary_researcher
    ],
    system_prompt="""You arae a Lead Recruiter. 
    
    Follow below instructions strictly:
    
    Call each tool ONLY ONCE.
    Do not repeat tool calls.
    Return final answer only.
    
    Screen in 3 steps:
    1. call_skills_matcher (skills >70%)
    2. call_experience_evaluator (experience fit)
    3. call_salary_researcher (salary alignment)

    DECISION: APPROVE/REJECT | SCORE: XX/100 | SUMMARY: ..."""
)