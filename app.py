from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from dotenv import load_dotenv
import os, asyncio, re, json
import datetime


from tools import (
    get_user,
    get_timetable,
    get_additive_manufacturing_equipment,
    get_available_slots,
    add_booking,
    clear_worksheet,
    list_excel_files,
    read_log_files,
    write_session_log,
)

load_dotenv()


client = OpenAIChatCompletionClient(  #Main client
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



#TODO Make out commented out clients that use different models for easy testing later











validator_agent = AssistantAgent(
    name="validator_agent",
    model_client=client,
    tools=[],
    system_message="""
    You are an input validator. You will receive a message in this format:
    RULES: <what makes input valid>
    INPUT: <what the user typed>
    Reply ONLY with valid JSON — no markdown, no extra text:
    {"valid": true,  "value": "<cleaned value>"}
    {"valid": false, "reason": "<short explanation>"}
""",
    reflect_on_tool_use=False,
    model_client_stream=False,
)


async def validated_input(prompt: str, rules: str) -> str:
    """Checks Input() before sending to LLM. Repeats if invalid"""
    return input(prompt).strip()  ### DEBUGGING — bypasses validation ###ONLY FOR DEBUGGIND
    max_retries = 5
    for i in range(5):
        raw = input(prompt).strip()

        task = f"RULES: {rules}\nINPUT: {raw}"
        result = await run_agent(validator_agent, task)

        try:
            data = json.loads(clean_json_string(result))
        except json.JSONDecodeError:
            print("[Validator error — retrying]")
            continue

        if data.get("valid"):
            return data.get("value", raw)
        else:
            print(f"{data.get('reason', 'Invalid input.')}  Please try again.\n")

    print("Max retries reached — using last input")
    return raw














# Admin agent and function

admin_agent = AssistantAgent(
    name="admin_agent",
    model_client=client,
    tools=[list_excel_files, get_timetable, clear_worksheet, read_log_files],
    system_message="""
    You are an admin assistant for a manufacturing lab.
    Handle these requests:
    - "show files" / "what sheets" → call list_excel_files
    - "show timetable / schedule for X" → call list_excel_files first, then get_timetable
    - "clear sheet X" → confirm then call clear_worksheet
    - "show logs / summarize logs" → call read_log_files and summarize clearly

    Always confirm before clearing. Be concise.
    """,
    reflect_on_tool_use=True,
    model_client_stream=False,
)

async def run_admin_session() -> None:
    print("\n=== Admin Panel ===")
    print(f"You can ask the agent to check excel files and the timetable.\n The agent can also clear the timetable and read the logs to answer questions abot them")
    print("Type EXIT to quit.\n")
    while True:
        #user_input = input("Admin> ").strip()
        user_input = await validated_input(
            prompt="Admin> ",
            rules="must be related to administration, such as checking timetables, clearing sheets, quesitons related to logs. Reject anything unrelated."
        )
        if user_input.upper() == "EXIT":
            break
        response = await run_agent(admin_agent, user_input)
        print(f"\n{response}\n")


# User agents

planning_agent = AssistantAgent(
    name="planning_agent",
    model_client=client,
    tools=[get_additive_manufacturing_equipment],
    system_message="""
    You are a manufacturing planner.

    1. Call get_additive_manufacturing_equipment() with no arguments.
    2. Build a step-by-step plan based on available equipment.
    3. You MUST output ONLY valid JSON. Do not include markdown formatting, conversational text, or code blocks.
    4. IMPORTANT: All 'duration_hours' for items that require booking MUST be whole numbers (integers). Do not add decimals or buffer time (e.g., use 2, not 2.1).
    Format your response exactly as a JSON array of objects:
    [
      {
        "step": "Step 1",
        "tool": "Fusion 360",
        "action": "Design the part",
        "duration_hours": 2,
        "requires_booking": false
      }
    ]
    """,
    reflect_on_tool_use=True,
    model_client_stream=False,
)




def make_slot_picker_agent() -> AssistantAgent:
    return AssistantAgent(
        name="slot_picker_agent",
        model_client=client,
        tools=[],
        system_message="""
    You are given a list of available time slots and user preferences.
    Pick the ONE slot that best matches the preferences.
    Reply with ONLY valid JSON, no markdown, no extra text.

    If a suitable slot exists:
    {"status": "FOUND", "day": "tuesday", "start": "18:00", "end": "22:00"}

    If none match the preferences:
    {"status": "NONE"}
    """,
        reflect_on_tool_use=False,
        model_client_stream=False,
    )


# Helper functions



def find_file_and_sheet(tool_name: str, files: dict) -> tuple[str, str] | None:
    """Match a tool name to an Excel file and sheet using word overlap incase agent isn't exact."""
    tool_words = set(tool_name.lower().split())
    best_score = 0
    best_match = None
    for filename, sheets in files.items():
        for sheet in sheets:
            sheet_words = set(sheet.lower().split())
            score = len(tool_words & sheet_words)
            if score > best_score:
                best_score = score
                best_match = (filename, sheet)
    return best_match if best_score > 0 else None



def get_tool_cost(tool_name: str, db: dict) -> int:
    """Get costs of any tools"""
    if not tool_name:
        return 0

    target_name = tool_name.strip().lower()

    for category in db.values():
        for tool in category:
            if tool['name'].lower() == target_name:
                return tool['cost']

    return 0

def clean_json_string(raw_text: str) -> str:
    """Helper to strip markdown code blocks"""
    text = raw_text.strip()
    # FIX: strip ```json or plain ``` fences
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

async def parse_plan(raw_plan: str) -> tuple[list[dict], int]:
    
    clean_text = clean_json_string(raw_plan)
    try:
        steps = json.loads(clean_text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return [], 0


    if not isinstance(steps, list):
        print(f"[ERROR] Expected JSON array but got {type(steps).__name__}")
        return [], 0

    total_cost = 0
    
    db = await get_additive_manufacturing_equipment()

    for s in steps:
        hourly_rate = get_tool_cost(s['tool'], db)
        s['cost'] = hourly_rate * s['duration_hours']
        total_cost += s['cost']

    return steps, total_cost

def parse_slot_result(slot_result: str) -> tuple[str, str, str] | None:
    """Parse the JSON slot dictionary."""
    clean_text = clean_json_string(slot_result)
    if not clean_text:
        print("[WARN] Timetable agent returned an empty response — will retry.")
        return None
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return None

    if data.get("status") == "FOUND":
        return data["day"], data["start"], data["end"]
        
    return None

def rebuild_plan_text(steps: list[dict]) -> str:
    lines = []
    for s in steps:
        step     = s.get('step','')
        tool     = s.get('tool','')
        action   = s.get('action','')
        duration = s.get('duration_hours', 0)
        booking  = "Yes" if s.get('requires_booking') else "No"

        lines.append(
            f"{step}: {tool}\n{action}\n"
            f"Duration: {duration}h - Booking required: {booking}\n"
        )

    return "\n".join(lines)
async def format_plan_with_cost(steps: list[dict]) -> str:
    lines = []
    total = 0
    db = await get_additive_manufacturing_equipment()

    for s in steps:
        tool     = s.get('tool', '')
        step     = s.get('step', '')
        duration = s.get('duration_hours', 0)
        cost     = s.get('cost', 0)
        action   = s.get('action', '')

        hourly     = get_tool_cost(tool, db)
        cost_str   = f"@ €{hourly}/h = €{cost}" if cost > 0 else "(free)"
        booking    = " [BOOKING REQUIRED]" if s.get('requires_booking') else ""
        action_str = f" {action}" if action else ""

        lines.append(f"  {step}: {tool} — {duration}h {cost_str}{booking}\n{action_str}\n")
        total += cost

    lines.append(f"\n  Total cost: €{total}")
    return "\n".join(lines)


async def run_agent(agent: AssistantAgent, task: str) -> str:
    '''Helper to run agents'''
    result = await agent.run(task=task)

    for msg in reversed(result.messages):
        # Skip the user prompt and any non-string or empty content
        if getattr(msg, 'source', '') == 'user':
            continue
        if isinstance(msg.content, str) and msg.content.strip():
            return msg.content

    return ""



# Plan with user-editable durations 

MAX_PLAN_ATTEMPTS = 5  

async def planning_phase(user_request: str) -> tuple[str, list[dict], int]:
    """
    Generate plan, allow duration edits, loop until APPROVED.
    Returns (plan_text, steps, total_cost).
    """
    db = await get_additive_manufacturing_equipment()

    task  = user_request
    steps = []
    attempts = 0  

    while True:
        if not steps:
            if attempts >= MAX_PLAN_ATTEMPTS:
                raise RuntimeError(
                    f"Planning agent failed to return a valid JSON plan after {MAX_PLAN_ATTEMPTS} attempts."
                )
            attempts += 1

            raw_plan = await run_agent(planning_agent, task)
            steps, total_cost = await parse_plan(raw_plan)
            if not steps:
                print("\n[No steps found in plan — retrying...]")
                task = f"{user_request} — please provide a detailed step-by-step plan. You MUST format output as JSON."
                continue
        else:
            attempts = 0 
            total_cost = sum(s['cost'] for s in steps)

        print("PROPOSED PLAN:")
        print(await format_plan_with_cost(steps))
        print("="*60)
        print("\nOptions:")
        print("  APPROVED  — accept and proceed")
        print("  EDIT      — change a step duration")
        print("  <text>    — request changes to the plan")


        user_input = await validated_input(
            prompt="\nYour choice: ",
            rules="must be approved, edit or asking for something that involves chagind a manifcaturing plan."
        )

        if user_input.upper() == "APPROVED":
            plan_text = rebuild_plan_text(steps)
            return plan_text, steps, total_cost

        elif user_input.upper() == "EDIT":
            print("\n" + "="*60)

            print("\nCurrent steps:")
            for i, s in enumerate(steps):
                print(f"  {i+1}. {s['step']}: {s['tool']} — {s['duration_hours']}h")
            try:
                idx     = int(input("Step number to edit: ").strip()) - 1
                new_dur = int(input(f"New duration for '{steps[idx]['tool']}' (hours): ").strip())
                # if new_dur < 1:
                #     print("Duration must be at least 1 hour.")
                #     continue
                steps[idx]['duration_hours'] = new_dur

                steps[idx]['cost'] = get_tool_cost(steps[idx]['tool'], db) * new_dur
                print(f"\n Updated {steps[idx]['step']} to {new_dur}h")
            except (ValueError, IndexError):
                print("Invalid input — no changes made.")

        else:
            task  = (
                f"Revise the plan based on this feedback: {user_input}\n"
                f"Original request: {user_request}\n"
                f"You MUST output ONLY valid JSON format."
            )
            steps = []

#  Discussion

async def discussion_phase(approved_plan: str, steps: list[dict]) -> str:
    """
    Simple terminal chat for the user to state booking preferences.
    No agent involved — avoids the stuck-in-discussion bug entirely.
    Returns the user's raw preference text.

    The user types CONFIRM (or just presses Enter) to proceed.
    """
    print("\n" + "="*60)
    print(f"BOOKING PREFERENCES\n")
    print("Explain any preferences for scheduling, or press Enter to skip.")
    print("Examples: 'after 6pm'  |  'Tuesday for the printer'  |  'avoid Mondays'")
    print("Type CONFIRM or press Enter when ready.\n")

    preferences_lines = []

    while True:
        user_input=await validated_input(
            prompt="Preference (or CONFIRM): ",
            rules="must be related to a timing or day preference (in the noon or avoid mondays etc.). or empty or CONFIRM. Reject anything unrelated")
        #user_input = input("Preference (or CONFIRM): ").strip()
        if user_input.upper() == "CONFIRM" or user_input == "":
            break
        preferences_lines.append(user_input)
        print("   Noted.")

    preferences_text = " ".join(preferences_lines)
    print(f"\n[DEBUG] discussion_phase: preferences = '{preferences_text}'")
    print(" Proceeding to booking.\n")
    return preferences_text

# Booking

async def booking_phase(
    steps: list[dict],
    preferences_text: str,
    username: str,
) -> tuple[list[dict], list[str], bool]:

    bookings_log = []
    skipped      = []
    terminated   = False
    current_prefs = preferences_text

    # Fetch file/sheet mapping once for the whole session
    files = await list_excel_files()

    for item in steps:
        if not item.get('requires_booking'):
            continue

        step     = item['step']
        tool     = item['tool']
        duration = item['duration_hours']

        print(f"\n{'='*60}")
        print(f"Booking — {step}: {tool}  ({duration}h)")

        # Resolve file and sheet directly in Python — no agent needed
        match = find_file_and_sheet(tool, files)
        if not match:
            print(f"[ERROR] Could not match '{tool}' to any Excel sheet — skipping.")
            skipped.append(tool)
            continue
        file_name, sheet_name = match

        local_prefs = current_prefs
        booked      = False

        while not booked:

            slots_raw = await get_available_slots(file_name, sheet_name, duration)

            if slots_raw == "NO_SLOTS_AVAILABLE":
                print(f"\nNo {duration}h slot available for {tool}.")
                print("Options:  SKIP  |  CANCEL  |  <new preference to retry>")
                #choice = input("Choice: ").strip()
                choice=await validated_input(
                            prompt="Choice: ",
                            rules="must be related to a timing preference for booing or skip or cancel. Reject anything unrelated")

                            
                if choice.upper() == "CANCEL":
                    terminated = True
                    return bookings_log, skipped, terminated
                elif choice.upper() == "SKIP":
                    skipped.append(tool)
                    break
                else:
                    local_prefs = choice
                    continue

            pref_clause = f"User preferences: {local_prefs}." if local_prefs else "No preferences — pick the first available slot."
            pick_task = (
                f"Available {duration}-hour slots for '{tool}':  \n{slots_raw}\n\n"
                f"{pref_clause}"
            )
            pick_result = await run_agent(make_slot_picker_agent(), pick_task)
            parsed = parse_slot_result(pick_result)

            if not parsed:
                print(f"\nCould not select a slot for {tool}.")
                print("Options:  SKIP  |  CANCEL  |  <new preference to retry>")
#                choice = input("Choice: ").strip()
                choice=await validated_input(
                                prompt="Choice: ",
                                rules="must be related to a timing preference for booing or skip or cancel. Reject anything unrelated")

                if choice.upper() == "CANCEL":
                    terminated = True
                    return bookings_log, skipped, terminated
                elif choice.upper() == "SKIP":
                    skipped.append(tool)
                    break
                else:
                    local_prefs = choice
                    continue

            slot_day, slot_start, slot_end = parsed

            print(f"\nProposed: {tool}  |  {slot_day.capitalize()} {slot_start}-{slot_end}  ({duration}h)")
            confirm = input("Accept? (YES / NO / SKIP / CANCEL): ").strip().upper()

            if confirm == "YES":

                result = await add_booking(slot_day, slot_start, slot_end, username, file_name, sheet_name)
                if result == "Success":
                    print(f" Booked {tool} — {slot_day.capitalize()} {slot_start}-{slot_end}")
                    bookings_log.append({
                        'tool':       tool,
                        'day':        slot_day,
                        'start_time': slot_start,
                        'end_time':   slot_end,
                        'status':     'booked',
                    })
                    booked = True
                    current_prefs = local_prefs
                else:
                    print(f"[WARN] {result} — searching for another slot.")

            elif confirm == "NO":
                new_pref = input("Describe what slot you want instead (or Enter to retry same): ").strip()
                if new_pref:
                    local_prefs = new_pref

            elif confirm == "SKIP":
                skipped.append(tool)
                break

            elif confirm == "CANCEL":
                terminated = True
                return bookings_log, skipped, terminated

    return bookings_log, skipped, terminated


# User Session


async def run_user_session(username: str) -> None:

    hour = datetime.datetime.now().hour

    if 5 <= hour < 12:
        print(f"\n=== Good morning, {username}! ===\n")
    elif 12 <= hour < 18:
        print(f"\n=== Good afternoon, {username}! ===\n")
    else:
        print(f"\n=== Good evening, {username}! ===\n")



    user_request = input("What would you like me to plan? ").strip()

    print("\n" + "="*60)
    print("\n--- Generating Plan ---")


    approved_plan, steps, total_cost = await planning_phase(user_request)
    print(f"\n Plan approved. Total cost: €{total_cost}")

    needs_booking = [s for s in steps if s.get('requires_booking')]
    if needs_booking:
        print(f"\nSteps requiring booking ({len(needs_booking)}):")
        for s in needs_booking:
            print(f"  {s['step']}: {s['tool']} — {s['duration_hours']}h")

    preferences_text = await discussion_phase(approved_plan, steps)

    if not needs_booking:
        print("\nNo bookings required — all tools are freely available.")
        bookings_log, skipped, terminated = [], [], False
    else:
        bookings_log, skipped, terminated = await booking_phase(
            steps, preferences_text, username
        )

    skipped_cost = sum(s['cost'] for s in steps if s.get('tool') in skipped)
    total_cost -= skipped_cost
    if skipped_cost:
        print(f"Skipped bookings deducted: -€{skipped_cost}  →  Final cost: €{total_cost}")

    log_path = write_session_log(
        username=username,
        user_request=user_request,
        approved_plan=approved_plan,
        bookings=bookings_log,
        total_cost=total_cost,
        skipped=skipped,
    )
    print(f"\n=== Session complete. Log: {log_path} ===")



async def main() -> None:
    print("=== Manufacturing Booking System ===")
    user = get_user()
    if user == "admin":
        await run_admin_session()
    else:
        await run_user_session(user)

if __name__ == "__main__":
    asyncio.run(main())