import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
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
  

def say_hello(name: str = "there") -> str:
    """Provides a simple greeting, optionally addressing the user by name.

    Args:
        name (str, optional): The name of the person to greet. Defaults to "there".

    Returns:
        str: A friendly greeting message.
    """
    print(f"--- Tool: say_hello called with name: {name} ---")
    return f"Hello, {name}!"



def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation."""
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."

print("Greeting and Farewell tools defined.")


# Optional self-test
# print(say_hello("Alice"))
# print(say_goodbye())

greeting_agent = None


try:
    greeting_agent = Agent(
        # Using a potentially different/cheaper model for a simple task
        model=LiteLlm(model=MODEL_GPT_4O),
        name="greeting_agent",
        instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
                    "Use the 'say_hello' tool to generate the greeting. "
                    "If the user provides their name, make sure to pass it to the tool. "
                    "Do not engage in any other conversation or tasks.",
        description="Handles simple greetings and hellos using the 'say_hello' tool.", # Crucial for delegation
        tools=[say_hello],
    )
    print(f"Greeting Agent created using model")
except Exception as e:
    print(f"Could not create Greeting agent. Check API Key (). Error: {e}")

# --- Farewell Agent ---
farewell_agent = None


try:
    farewell_agent = Agent(
        # Can use the same or a different model
        model=LiteLlm(model=MODEL_GPT_4O), # Sticking with GPT for this example
        name="farewell_agent",
        instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
                    "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation "
                    "(e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
                    "Do not perform any other actions.",
        description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.", # Crucial for delegation
        tools=[say_goodbye],
    )
    print(f"Farewell agent created' created using model ''.")
except Exception as e:
    print(f" Could not create Farewell agent. Check API Key (). Error: {e}")
    


if greeting_agent and farewell_agent and 'get_weather' in globals():
    root_agent_model = AGENT_MODEL 
    root_agent = Agent(
        name="weather_agent_v2", # Give it a new version name
        model=root_agent_model,
        description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
        instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
                    "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
                    "You have specialized sub-agents: "
                    "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
                    "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
                    "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
                    "If it's a weather request, handle it yourself using 'get_weather'. "
                    "For anything else, respond appropriately or state you cannot handle it.",
        tools=[get_weather], # Root agent still needs the weather tool for its core task
        # Key change: Link the sub-agents here!
        sub_agents=[greeting_agent, farewell_agent]
    )
    print(f"Root Agent '{root_agent.name}' created using model '{root_agent_model}' with sub-agents: {[sa.name for sa in root_agent.sub_agents]}")

    # --- IMPORTANT: Create a Runner for this Root Agent ---
    # This runner will manage the multi-agent flow starting from root_agent.
    runner_root = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service # Re-use the same session service
    )
    print(f"✅ Runner created for the root agent '{runner_root.agent.name}'.")

else:
    print("❌ Cannot create root agent because one or more sub-agents failed to initialize or 'get_weather' tool is missing.")
    if not greeting_agent: print(" - Greeting Agent is missing.")
    if not farewell_agent: print(" - Farewell Agent is missing.")
    if 'get_weather' not in globals(): print(" - get_weather function is missing.")
    

# if runner_root:
#   async def run_team_conversation():
#       print("\n--- Testing Agent Team Delegation ---")
#       # Always interact via the root agent's runner
#       await call_specific_agent_async(runner_root, "Hello there!")
#       await call_specific_agent_async(runner_root, "What is the weather in New York?")
#       await call_specific_agent_async(runner_root, "Thanks, bye!")

#   # Execute the conversation
#   # Note: This may require API keys for the models used by root and sub-agents!
#   print("running the subagents team conversation")
#   asyncio.run(run_team_conversation()); 
# else:
#   print("\n⚠️ Skipping agent team conversation as the root agent runner ('runner_root') is not available.")
  
session_service_stateful = InMemorySessionService()
print("✅ New InMemorySessionService created for state demonstration.")

# Define a NEW session ID for this part of the tutorial
SESSION_ID_STATEFUL = "session_state_demo_001"
USER_ID_STATEFUL = "user_state_demo"


# Define initial state data - user prefers Celsius initially
initial_state = {
    "user_preference_temperature_unit": "Celsius"
}

# Create the session, providing the initial state
session_stateful = session_service_stateful.create_session(
    app_name=APP_NAME, # Use the consistent app name
    user_id=USER_ID_STATEFUL,
    session_id=SESSION_ID_STATEFUL,
    state=initial_state # <<< Initialize state during creation
)
print(f"✅ Session '{SESSION_ID_STATEFUL}' created for user '{USER_ID_STATEFUL}'.")

# Verify the initial state was set correctly
retrieved_session = session_service_stateful.get_session(APP_NAME, USER_ID_STATEFUL, SESSION_ID_STATEFUL)
print("\n--- Initial Session State ---")
if retrieved_session:
    print(retrieved_session.state)
else:
    print("Error: Could not retrieve session.")
    
def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
    """Retrieves weather, converts temp unit based on session state."""
    print(f"--- Tool: get_weather_stateful called for {city} (SessionID: {tool_context.session.id}) ---")

    # --- Read preference from state ---
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius") # Default to Celsius
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")

    city_normalized = city.lower().replace(" ", "")

    # Mock weather data (always stored in Celsius internally)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]

        # Format temperature based on state preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9/5) + 32 # Calculate Fahrenheit
            temp_unit = "°F"
        else: # Default to Celsius
            temp_value = temp_c
            temp_unit = "°C"

        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        result = {"status": "success", "report": report}
        print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")

        # Example of writing back to state (optional for this tool)
        tool_context.state["last_city_checked_stateful"] = city
        print(f"--- Tool: Updated state 'last_city_checked_stateful': {city} ---")

        return result
    else:
        # Handle city not found
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")
        return {"status": "error", "error_message": error_msg}

print("✅ State-aware 'get_weather_stateful' tool defined.")




greeting_agent = None
try:
    greeting_agent = Agent(
        model=LiteLlm(model=MODEL_GPT_4O),
        name="greeting_agent",
        instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting using the 'say_hello' tool. Do nothing else.",
        description="Handles simple greetings and hellos using the 'say_hello' tool.",
        tools=[say_hello],
    )
    print(f"✅ Agent '{greeting_agent.name}' redefined.")
except Exception as e:
    print(f"❌ Could not redefine Greeting agent. Error: {e}")

# --- Redefine Farewell Agent (from Step 3) ---
farewell_agent = None
try:
    farewell_agent = Agent(
        model=LiteLlm(model=MODEL_GPT_4O),
        name="farewell_agent",
        instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message using the 'say_goodbye' tool. Do not perform any other actions.",
        description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.",
        tools=[say_goodbye],
    )
    print(f"✅ Agent '{farewell_agent.name}' redefined.")
except Exception as e:
    print(f"❌ Could not redefine Farewell agent. Error: {e}")

# --- Define the Updated Root Agent ---
root_agent_stateful = None
runner_root_stateful = None # Initialize runner

# Check prerequisites before creating the root agent
if greeting_agent and farewell_agent and 'get_weather_stateful' in globals():

    root_agent_model = AGENT_MODEL # Choose orchestration model

    root_agent_stateful = Agent(
        name="weather_agent_v4_stateful", # New version name
        model=root_agent_model,
        description="Main agent: Provides weather (state-aware unit), delegates greetings/farewells, saves report to state.",
        instruction="You are the main Weather Agent. Your job is to provide weather using 'get_weather_stateful'. "
                    "The tool will format the temperature based on user preference stored in state. "
                    "Delegate simple greetings to 'greeting_agent' and farewells to 'farewell_agent'. "
                    "Handle only weather requests, greetings, and farewells.",
        tools=[get_weather_stateful], # Use the state-aware tool
        sub_agents=[greeting_agent, farewell_agent], # Include sub-agents
        output_key="last_weather_report" # <<< Auto-save agent's final weather response
    )
    print(f"✅ Root Agent '{root_agent_stateful.name}' created using stateful tool and output_key.")

    # --- Create Runner for this Root Agent & NEW Session Service ---
    runner_root_stateful = Runner(
        agent=root_agent_stateful,
        app_name=APP_NAME,
        session_service=session_service_stateful # Use the NEW stateful session service
    )
    print(f"✅ Runner created for stateful root agent '{runner_root_stateful.agent.name}' using stateful session service.")

else:
    print("❌ Cannot create stateful root agent. Prerequisites missing.")
    if not greeting_agent: print(" - greeting_agent definition missing.")
    if not farewell_agent: print(" - farewell_agent definition missing.")
    if 'get_weather_stateful' not in globals(): print(" - get_weather_stateful tool missing.")
    
if runner_root_stateful:
  async def run_stateful_conversation():
      print("\n--- Testing State: Temp Unit Conversion & output_key ---")

      # Define interaction helper using the stateful runner and session details
      interaction_func = lambda query: call_specific_agent_async(runner_root_stateful, query)
                                             # ^ Using the function from Step 2
                                             # Ensure USER_ID_STATEFUL, SESSION_ID_STATEFUL are used if modifying the helper

      # 1. Check weather (Uses initial state: Celsius)
      await interaction_func("What's the weather in London?")

      # 2. Manually update state preference to Fahrenheit
      print("\n--- Manually Updating State: Setting unit to Fahrenheit ---")
      current_session = session_service_stateful.get_session(APP_NAME, USER_ID_STATEFUL, SESSION_ID_STATEFUL)
      if current_session:
          current_session.state["user_preference_temperature_unit"] = "Fahrenheit"
          print(f"--- Session state updated. Current 'user_preference_temperature_unit': {current_session.state['user_preference_temperature_unit']} ---")
      else:
          print("--- Error: Could not retrieve session to update state. ---")

      # 3. Check weather again (Tool should now use Fahrenheit)
      # This will also update 'last_weather_report' via output_key
      await interaction_func("Tell me the weather in New York.")

      # 4. Test basic delegation (should still work)
      await interaction_func("Hi!")


  # Execute the conversation
  asyncio.run(run_stateful_conversation()); 

  # Inspect final session state after the conversation
  final_session = session_service_stateful.get_session(APP_NAME, USER_ID_STATEFUL, SESSION_ID_STATEFUL)
  
  
  if final_session:
      print("\n--- Final Session State ---")
      print(f"Preference: {final_session.state.get('user_preference_temperature_unit')}")
      print(f"Last Weather Report (from output_key): {final_session.state.get('last_weather_report')}")
      print(f"Last City Checked (by tool): {final_session.state.get('last_city_checked_stateful')}")
      # Print full state for detailed view
      # print(f"Full State: {final_session.state}")
  else:
      print("\n❌ Error: Could not retrieve final session state.")

else:
  print("\n⚠️ Skipping state test conversation. Stateful root agent runner ('runner_root_stateful') is not available.")    

# if __name__ == "__main__":
#     print("calling the agent file")
#     asyncio.run(run_multi_model_conversation())
  