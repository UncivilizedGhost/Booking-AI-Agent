from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from dotenv import load_dotenv
import os, asyncio

load_dotenv()

def make_client():
    return OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ['GOOGLE_API_KEY'],
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="gemini-2.5-flash",
            structured_output=True
        )
    )

from tools import get_timetable, add_booking, get_additive_manufacturing_equipment

# ── Mock check_tool_requirements ──────────────────────────────────────────────
async def check_tool_requirements(tool_name: str) -> str:
    """
    Check whether a tool or machine needs to be booked or is always available.

    Parameters
    ----------
    tool_name : str
        The name of the tool or machine to check.

    Returns
    -------
    str
        "always_available" or "requires_booking"
    """
    always_available = [
        "calipers", "micrometers", "sandpaper", "support removal tools",
        "cura", "prusaslicer", "preform", "fusion 360", "solidworks", "autocad",
        "paints", "files", "flush snips", "pliers"
    ]
    for a in always_available:
        if a in tool_name.lower():
            return "always_available"
    return "requires_booking"

# ── Agent definitions ─────────────────────────────────────────────────────────

planning_agent = AssistantAgent(
    name="planning_agent",
    model_client=make_client(),
    tools=[get_additive_manufacturing_equipment],
    system_message="""
    You are a manufacturing planner.
    1. Call get_additive_manufacturing_equipment with no arguments to get all equipment.
    2. Based on the user request and available equipment, create a numbered step-by-step plan.
       Format each step exactly as:
       Step N: [Tool Name] - [What you will do with it]
    3. Present the plan and wait for approval.
    Do not check availability. Do not book anything.
    """,
    reflect_on_tool_use=True,
    model_client_stream=False,
)

requirement_checker = AssistantAgent(
    name="requirement_checker",
    model_client=make_client(),
    tools=[check_tool_requirements],
    system_message="""
    You are a tool availability checker.
    You will receive an approved manufacturing plan.
    For every step in the plan, call check_tool_requirements with the tool name.
    After checking all steps, output a summary in this exact format:
    
    REQUIREMENTS:
    Step N: [Tool Name] - always_available
    Step N: [Tool Name] - requires_booking
    
    End your message with REQUIREMENTS_DONE.
    Do not book anything.
    """,
    reflect_on_tool_use=True,
    model_client_stream=False,
)

timetable_agent = AssistantAgent(
    name="timetable_agent",
    model_client=make_client(),
    tools=[get_timetable],
    system_message="""
    You are a schedule reader.
    1. Call get_timetable to get the current schedule.
    2. Find the earliest slot with enough consecutive free (NaN) 30-min blocks 
       for the requested duration on the requested day (or any day if not specified).
    3. Respond with ONLY one of these:
       SLOT_FOUND: <day> from <HH:MM> to <HH:MM>
       NO_SLOT_AVAILABLE
    Do not book anything.
    """,
    reflect_on_tool_use=True,
    model_client_stream=False,
)

booking_agent = AssistantAgent(
    name="booking_agent",
    model_client=make_client(),
    tools=[add_booking],
    system_message="""
    You are a booking agent.
    You will be given a day, start_time, end_time, and a name to book.
    Call add_booking with:
    - day: lowercase day name (e.g. "monday")
    - start_time: 24-hour format (e.g. "10:00")
    - end_time: 24-hour format (e.g. "12:00")
    - name: the tool name being reserved
    Report: BOOKING_SUCCESS or BOOKING_FAILED
    """,
    reflect_on_tool_use=True,
    model_client_stream=False,
)

# ── Helper: run a single agent on a task and return its last message ──────────

async def run_agent(agent: AssistantAgent, task: str) -> str:
    team = RoundRobinGroupChat(participants=[agent], max_turns=1)
    result = await Console(team.run_stream(task=task))
    for msg in reversed(result.messages):
        if hasattr(msg, 'content') and msg.content and msg.source != 'user':
            print("######################")
            print(type(msg.content))
            print((msg.content))
            print("######################")

            return msg.content
    return ""

# ── Phase 1: Plan generation loop until APPROVED ─────────────────────────────

async def planning_phase(user_request: str) -> str:
    task = user_request
    while True:
        plan = await run_agent(planning_agent, task)
        print("\n" + "="*60)
        user_input = input("Type APPROVED to proceed, or describe changes: ").strip()
        if user_input.upper() == "APPROVED":
            return plan
        else:
            task = (
                f"The user wants the following changes to the plan: {user_input}\n"
                f"Original request: {user_request}\n"
                f"Please revise the plan accordingly."
            )

# ── Phase 2: Requirements check ───────────────────────────────────────────────

async def requirements_phase(approved_plan: str) -> list[dict]:
    """Returns a list of steps that require booking: [{step, tool}]"""
    print("\n--- Checking tool requirements ---")
    result = await run_agent(
        requirement_checker,
        f"Check tool requirements for every step in this plan:\n{approved_plan}"
    )

    needs_booking = []
    for line in result.split('\n'):
        line = line.strip()
        if 'requires_booking' in line and line.startswith('Step'):
            # Parse "Step N: Tool Name - requires_booking"
            try:
                step_part, rest = line.split(':', 1)
                tool_name = rest.replace('- requires_booking', '').strip()
                needs_booking.append({
                    'step': step_part.strip(),
                    'tool': tool_name
                })
            except ValueError:
                continue

    return needs_booking

# ── Phase 3: Booking loop ─────────────────────────────────────────────────────

async def booking_phase(needs_booking: list[dict]) -> None:
    for item in needs_booking:
        step = item['step']
        tool = item['tool']
        print(f"\n--- Finding slot for {step}: {tool} ---")

        booked = False
        day_hint = ""

        while not booked:
            task = f"Find the earliest free 2-hour slot for '{tool}'"
            if day_hint:
                task += f" on {day_hint}"

            slot_result = await run_agent(timetable_agent, task)

            if slot_result.startswith("SLOT_FOUND"):
                # e.g. "SLOT_FOUND: tuesday from 10:00 to 12:00"
                slot_info = slot_result.replace("SLOT_FOUND:", "").strip()
                print(f"Slot found: {slot_info}")

                booking_result = await run_agent(
                    booking_agent,
                    f"Book this slot. Name: '{tool}'. Slot: {slot_info}"
                )

                if "BOOKING_SUCCESS" in booking_result:
                    print(f"✓ Booked {tool}: {slot_info}")
                    booked = True
                else:
                    print(f"Booking failed for {tool}.")
                    user_choice = input(
                        "Options:\n"
                        "  1. Try a different day (type day name e.g. 'monday')\n"
                        "  2. Skip this step (SKIP)\n"
                        "  3. Proceed without booking (PROCEED)\n"
                        "  4. Cancel the whole plan (CANCEL)\n"
                        "Your choice: "
                    ).strip().upper()

                    if user_choice == "CANCEL":
                        print("Plan cancelled.")
                        return
                    elif user_choice == "SKIP":
                        print(f"Skipping {tool}.")
                        break
                    elif user_choice == "PROCEED":
                        print(f"Proceeding without booking {tool}.")
                        break
                    else:
                        day_hint = user_choice.lower()

            else:
                # NO_SLOT_AVAILABLE
                print(f"No available slot found for {tool}.")
                user_choice = input(
                    "Options:\n"
                    "  1. Try a different day (type day name e.g. 'monday')\n"
                    "  2. Skip this step (SKIP)\n"
                    "  3. Proceed without booking (PROCEED)\n"
                    "  4. Cancel the whole plan (CANCEL)\n"
                    "Your choice: "
                ).strip().upper()

                if user_choice == "CANCEL":
                    print("Plan cancelled.")
                    return
                elif user_choice == "SKIP":
                    print(f"Skipping {tool}.")
                    break
                elif user_choice == "PROCEED":
                    print(f"Proceeding without booking {tool}.")
                    break
                else:
                    day_hint = user_choice.lower()

# ── Main ──────────────────────────────────────────────────────────────────────

import os
import hashlib

def get_user():
    attempts = 0

    while attempts < 3:
        username = input("Enter username: ")

        if username == "admin":
            password = input("Enter password: ")

            # Hash the entered password using SHA-256
            hashed_input = hashlib.sha256(password.encode()).hexdigest()
            print(hashed_input)

            # Compare with stored hash in environment variable
            if hashed_input == os.environ["PASSWORD"]:
                return "admin"
            else:
                print("Incorrect password.")
                attempts += 1
        else:
            return username

    # After 3 failed attempts, ask for username one last time (non-admin)
    return input("Enter username: ")


async def main() -> None:
   
   user= get_user()

    if user is not "admin":

        # print("=== Additive Manufacturing Booking System ===\n")
        # user_request = input("What would you like to manufacture? ").strip()

        # # Phase 1: Generate and approve plan
        # print("\n--- Generating manufacturing plan ---")
        # approved_plan = await planning_phase(user_request)
        #Also give cost of planning
        # print("\n✓ Plan approved.")

        ## pahs 2: 
        #For all tools that need booking book times

        # print("\n=== Workflow complete ===")

asyncio.run(main())