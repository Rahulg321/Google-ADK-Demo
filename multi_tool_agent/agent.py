import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

import asyncio
from google.genai import types 

from google.adk.models.lite_llm import LiteLlm

from dotenv import load_dotenv

load_dotenv()


def get_weather(city:str) -> dict:
    """ Retrieves the current weather report for a specified city
    
    Args:
        city(str) : The name of the city (e.g.,  "New York", "London", "Tokyo")
    
    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.

    """
    print(f"-----Tool: get weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "")
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }
    
    
    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


AGENT_MODEL = "gemini-2.0-flash-exp" 



weather_agent = Agent(
    name="weather_agent_v1",
    model=AGENT_MODEL, # Specifies the underlying LLM
    description="Provides weather information for specific cities.", # Crucial for delegation later
    instruction="You are a helpful weather assistant. Your primary goal is to provide current weather reports. "
                "When the user asks for the weather in a specific city, "
                "you MUST use the 'get_weather' tool to find the information. "
                "Analyze the tool's response: if the status is 'error', inform the user politely about the error message. "
                "If the status is 'success', present the weather 'report' clearly and concisely to the user. "
                "Only use the tool when a city is mentioned for a weather request.",
    tools=[get_weather], # Make the tool available to this agent
)




print(f"Agent '{weather_agent.name}' created using model '{AGENT_MODEL}'.")


# --- Session Management ---
# Key Concept: SessionService stores conversation history & state.
# InMemorySessionService is simple, non-persistent storage for this tutorial.
session_service = InMemorySessionService()

# Define constants for identifying the interaction context
APP_NAME = "weather_tutorial_app"
USER_ID = "user_1"
SESSION_ID = "session_001" # Using a fixed ID for simplicity

# Create the specific session where the conversation will happen
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)



print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")


# --- Runner ---
# Key Concept: Runner orchestrates the agent execution loop.
runner = Runner(
    agent=weather_agent, # The agent we want to run
    app_name=APP_NAME,   # Associates runs with our app
    session_service=session_service # Uses our session manager
)



print(f"Runner created for agent '{runner.agent.name}'.")



async def call_agent_async(query: str):
  """Sends a query to the agent and prints the final response."""
  print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
      # You can uncomment the line below to see *all* events during execution
    #   print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

  print(f"<<< Agent Response: {final_response_text}")
  



# We need an async function to await our interaction helper
async def run_conversation():
    await call_agent_async("What is the weather like in London?")
    await call_agent_async("How about Paris?") # Expecting the tool's error message
    await call_agent_async("Tell me the weather in New York")




MODEL_GPT_4O="gpt-4o"
MODEL_CLAUDE_SONNET="claude-3-5-sonnet-latest"

# --- Agent using GPT-4o ---
weather_agent_gpt = None # Initialize to None
runner_gpt = None      # Initialize runner to None
try:
    weather_agent_gpt = Agent(
        name="weather_agent_gpt",
        # Key change: Wrap the LiteLLM model identifier
        model=LiteLlm(model=MODEL_GPT_4O),
        description="Provides weather information (using GPT-4o).",
        instruction="You are a helpful weather assistant powered by GPT-4o. "
                    "Use the 'get_weather' tool for city weather requests. "
                    "Clearly present successful reports or polite error messages based on the tool's output status.",
        tools=[get_weather], # Re-use the same tool
    )
    print(f"Agent '{weather_agent_gpt.name}' created using model '{MODEL_GPT_4O}'.")
    # Create a runner specific to this agent
    runner_gpt = Runner(
        agent=weather_agent_gpt,
        app_name=APP_NAME,       # Use the same app name
        session_service=session_service # Re-use the same session service
        )
    print(f"Runner created for agent '{runner_gpt.agent.name}'.")

except Exception as e:
    print(f"❌ Could not create GPT agent '{MODEL_GPT_4O}'. Check API Key and model name. Error: {e}")

# --- Agent using Claude Sonnet ---
weather_agent_claude = None # Initialize to None
runner_claude = None      # Initialize runner to None



try:
    weather_agent_claude = Agent(
        name="weather_agent_claude",
        # Key change: Wrap the LiteLLM model identifier
        model=LiteLlm(model=MODEL_CLAUDE_SONNET),
        description="Provides weather information (using Claude Sonnet).",
        instruction="You are a helpful weather assistant powered by Claude Sonnet. "
                    "Use the 'get_weather' tool for city weather requests. "
                    "Analyze the tool's dictionary output ('status', 'report'/'error_message'). "
                    "Clearly present successful reports or polite error messages.",
        tools=[get_weather], # Re-use the same tool
    )
    print(f"Agent '{weather_agent_claude.name}' created using model '{MODEL_CLAUDE_SONNET}'.")
    # Create a runner specific to this agent
    runner_claude = Runner(
        agent=weather_agent_claude,
        app_name=APP_NAME,       # Use the same app name
        session_service=session_service # Re-use the same session service
        )
    print(f"Runner created for agent '{runner_claude.agent.name}'.")

except Exception as e:
    print(f"❌ Could not create Claude agent '{MODEL_CLAUDE_SONNET}'. Check API Key and model name. Error: {e}")
    


async def call_specific_agent_async(runner_instance: Runner, query: str):
  """Sends a query to the agent via the specified runner and prints the final response."""
  if not runner_instance:
      print(f"⚠️ Cannot run query '{query}'. Runner is not available (check agent creation and API keys).")
      return

  agent_name = runner_instance.agent.name
  print(f"\n>>> User Query to {agent_name}: {query}")

  content = types.Content(role='user', parts=[types.Part(text=query)])
  final_response_text = f"Agent {agent_name} did not produce a final response." # Default

  try:
    # Use the specific runner passed to the function
    async for event in runner_instance.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}") # Debugging
        if event.is_final_response():
            if event.content and event.content.parts:
               final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
               final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break # Exit loop once final response is found
  except Exception as e:
      # Catch potential errors during the run (e.g., API errors)
      print(f"❌ Error during run for agent {agent_name}: {e}")
      final_response_text = f"Error occurred while running agent {agent_name}."


  print(f"<<< {agent_name} Response: {final_response_text}")
  
  
  
# --- Example Usage ---
async def run_multi_model_conversation():
    print("\n--- Testing GPT Agent ---")
    # Check if the runner was created successfully before calling
    if runner_gpt:
      await call_specific_agent_async(runner_gpt, "What's the weather in Tokyo?")
    else:
      print("Skipping GPT test as runner is not available.")

    print("\n--- Testing Claude Agent ---")
    # Check if the runner was created successfully
    if runner_claude:
      await call_specific_agent_async(runner_claude, "Weather in London please.")
    else:
       print("Skipping Claude test as runner is not available.")

    print("\n--- Testing Original Gemini Agent ---")
    # Assuming 'runner' still holds the runner for the original Gemini agent (weather_agent_v1 from Step 1)
    # Ensure the 'runner' variable from Step 1 is accessible here.
    # If running steps independently, you might need to recreate the original runner.
    if 'runner' in globals() and runner:
         await call_specific_agent_async(runner, "How about New York?")
    else:
         print("Original Gemini agent runner ('runner') not found. Skipping test.")
  

if __name__ == "__main__":
    print("calling the agent file")
    asyncio.run(run_multi_model_conversation())
  