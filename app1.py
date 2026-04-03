from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from dotenv import load_dotenv
import os, asyncio, re, json

load_dotenv()

from tools1 import (
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


client = OpenAIChatCompletionClient(
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


async def run_agent(agent: AssistantAgent, task: str,max_turns:int=0) -> str:
    '''Helper function to run agents and get last message'''
    result = await agent.run(task=task)
    
    if result.messages:
        last_msg = result.messages[-1]
        return str(last_msg.content)
    
    return ""
# ═════════════════════════════════════════════════════════════════════════════
# Admin functions
# ═════════════════════════════════════════════════════════════════════════════

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
    '''Simple bot converstaion where it can do stuff'''
    print("\n=== Admin Panel ===")
    print("Commands: view files, show timetable, clear worksheets, read logs.")
    print("Type EXIT to quit.\n")
    while True:
        user_input = input("Admin> ").strip()
        if user_input.upper() == "EXIT":
            break
        response = await run_agent(admin_agent, user_input)
        print(f"\n{response}\n")

# ═════════════════════════════════════════════════════════════════════════════
# User agents
# ═════════════════════════════════════════════════════════════════════════════

planning_agent = AssistantAgent(
    name="planning_agent",
    model_client=client,
    tools=[get_additive_manufacturing_equipment],
    system_message="""
    You are a manufacturing planner.

    1. Call get_additive_manufacturing_equipment() with no arguments.
    2. Build a step-by-step plan based on available equipment.
    3. You MUST output ONLY valid JSON. Do not include markdown formatting, conversational text, or code blocks.
    4. IMPORTANT: All 'duration_hours' MUST be whole numbers (integers). Do not add decimals or buffer time (e.g., use 2, not 2.1).
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

timetable_agent = AssistantAgent(
    name="timetable_agent",
    model_client=client,
    tools=[list_excel_files, get_available_slots],
    system_message="""
    You are a slot finder. 
    1. Call list_excel_files to find the correct file and sheet.
    2. Call get_available_slots.
    3. Pick the ONE best slot matching user preferences.
    
    4. You MUST reply with ONLY valid JSON. No markdown, no extra text.
    
    If a slot is found, return exactly this format:
    {"status": "FOUND", "day": "tuesday", "start": "18:00", "end": "22:00"}
    
    If no slots exist, return:
    {"status": "NONE"}
    """,
    reflect_on_tool_use=True,
    model_client_stream=False,
)

booking_agent = AssistantAgent(
    name="booking_agent",
    model_client=client,
    tools=[list_excel_files, add_booking],
    system_message="""
    You are a booking agent.
    You will be given a tool name, username, day, start_time, and end_time.

    1. Call list_excel_files to find the correct file and worksheet for the tool.
    2. Call add_booking with:
       - day: lowercase day name (e.g. "monday")
       - start_time: HH:00 format (e.g. "18:00")
       - end_time: HH:00 format (e.g. "21:00")
       - name: the USERNAME (not the tool name)
       - file_name: matched filename
       - sheet_name: matched worksheet
    3. Reply with ONLY: BOOKING_SUCCESS or BOOKING_FAILED
    """,
    reflect_on_tool_use=True,
    model_client_stream=False,
)

# ═════════════════════════════════════════════════════════════════════════════
# Plan Parsing
# ═════════════════════════════════════════════════════════════════════════════


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
    """Removes code blocks if LLM doens't give proper output"""
    text = raw_text.strip()
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
        print(f"[ERROR] Expected a JSON array but got {type(steps).__name__}")
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
    """Rebuild plan after editing it."""
    lines = []
    for s in steps:
        booking = "Yes" if s.get('requires_booking') else "No"
        lines.append(
            f"{s['step']}: {s['tool']} - Use {s['tool']} - "
            f"Duration: {s['duration_hours']}h - Booking required: {booking}"
        )
    
    plan="\n".join(lines)

    return plan

async def format_plan_with_cost(steps: list[dict]) -> str:
    """Human-readable plan yo print"""
    lines = []

    total = 0
    
    db = await get_additive_manufacturing_equipment()
    
    for s in steps:
        hourly   = get_tool_cost(s['tool'], db)
        cost_str="(free)"
        if (s['cost'] > 0):
            cost_str = f"@ €{hourly}/h = €{s['cost']}"

        booking  = " [BOOKING REQUIRED]" if s.get('requires_booking') else ""
        lines.append(f"  {s['step']}: {s['tool']} — {s['duration_hours']}h {cost_str}{booking}")
        total += s['cost']
        
    lines.append(f"\n  Total cost: €{total}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# USER 
# ═════════════════════════════════════════════════════════════════════════════

# ── Phase 1: Plan with user-editable durations ────────────────────────────────

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

            raw_plan = await run_agent(planning_agent, task, max_turns=1)
            steps, total_cost = await parse_plan(raw_plan)
            if not steps:
                print("\n[No steps found in plan — retrying...]")
                task = f"{user_request} — please provide a detailed step-by-step plan. You MUST format output as JSON."
                continue
        else:
            attempts = 0  # FIX: reset counter once we have a valid plan
            total_cost = sum(s['cost'] for s in steps)

        print("\n" + "="*60)
        print("PROPOSED PLAN:")
        print(await format_plan_with_cost(steps))
        print("="*60)
        print("\nOptions:")
        print("  APPROVED  — accept and proceed")
        print("  EDIT      — change a step duration")
        print("  <text>    — request changes to the plan")

        user_input = input("\nYour choice: ").strip()

        if user_input.upper() == "APPROVED":
            plan_text = rebuild_plan_text(steps)
            return plan_text, steps, total_cost

        elif user_input.upper() == "EDIT":
            print("\nCurrent steps:")
            for i, s in enumerate(steps):
                print(f"  {i+1}. {s['step']}: {s['tool']} — {s['duration_hours']}h")
            try:
                idx     = int(input("Step number to edit: ").strip()) - 1
                new_dur = int(input(f"New duration for '{steps[idx]['tool']}' (hours): ").strip())
                if new_dur < 1:
                    print("Duration must be at least 1 hour.")
                    continue
                steps[idx]['duration_hours'] = new_dur

                steps[idx]['cost'] = get_tool_cost(steps[idx]['tool'], db) * new_dur
                print(f"\n Updated {steps[idx]['step']} to {new_dur}h")
            except (ValueError, IndexError):
                print("Invalid input - no changes made.")

        else:
            task  = (
                f"Revise the plan based on this feedback: {user_input}\n"
                f"Original request: {user_request}\n"
                f"You MUST output ONLY valid JSON format."
            )
            steps = []

# ── Phase 2: Discussion — pure Python input loop, no agent needed ─────────────

def discussion_phase(approved_plan: str, steps: list[dict]) -> str:
    """
    Simple terminal chat for the user to state booking preferences.
    No agent involved — avoids the stuck-in-discussion bug entirely.
    Returns the user's raw preference text.

    The user types CONFIRM (or just presses Enter) to proceed.
    """
    print("\n" + "="*60)
    print("BOOKING PREFERENCES")
    print("Explain any preferences for scheduling, or press Enter to skip.")
    print("Examples: 'after 6pm'  |  'Tuesday for the printer'  |  'avoid Mondays'")
    print("Type CONFIRM or press Enter when ready.\n")

    preferences_lines = []

    while True:
        user_input = input("Preference (or CONFIRM): ").strip()
        if user_input.upper() == "CONFIRM" or user_input == "":
            break
        preferences_lines.append(user_input)
        print("  ✓ Noted.")

    preferences_text = " ".join(preferences_lines)
    print(f"\n[DEBUG] discussion_phase: preferences = '{preferences_text}'")
    print("✓ Proceeding to booking.\n")
    return preferences_text

# ── Phase 3: Booking ──────────────────────────────────────────────────────────

MAX_BOOKING_ATTEMPTS = 3  # FIX: cap retries when booking_agent keeps failing

async def booking_phase(
    steps: list[dict],
    preferences_text: str,
    username: str,
) -> tuple[list[dict], list[str], bool]:
    
    bookings_log = []
    skipped      = []
    terminated   = False

    current_prefs = preferences_text

    for item in steps:
        if not item.get('requires_booking'):
            continue

        step     = item['step']
        tool     = item['tool']
        duration = item['duration_hours']

        print(f"\n{'='*60}")
        print(f"Booking — {step}: {tool}  ({duration}h)")

        local_prefs = current_prefs
        booked      = False
        booking_attempts = 0  

        while not booked:
            pref_clause = f" User preferences: {local_prefs}." if local_prefs else ""
            slot_task   = (
                f"Find the best {duration}-hour slot for '{tool}'.{pref_clause}"
                f" Pick the slot that best matches the preferences."
            )

            slot_result = await run_agent(timetable_agent, slot_task, max_turns=3)

            parsed = parse_slot_result(slot_result)
            
            if not parsed:
                print(f"\nNo {duration}h slot available for {tool}.")
                print("Options:  SKIP  |  CANCEL  |  <new preference to retry>")
                choice = input("Choice: ").strip()
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
                booking_attempts += 1
                booking_result = await run_agent(
                    booking_agent,
                    f"Book slot for tool '{tool}' on {slot_day} from {slot_start} to {slot_end}. Username: '{username}'.",
                    max_turns=2
                )
                if "BOOKING_SUCCESS" in booking_result:
                    print(f"Booked {tool} — {slot_day.capitalize()} {slot_start}-{slot_end}")
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
                    if booking_attempts >= MAX_BOOKING_ATTEMPTS:
                        print(f"[ERROR] Booking agent failed {MAX_BOOKING_ATTEMPTS} times for {tool} — skipping.")
                        skipped.append(tool)
                        break
                    print(f"[WARN] Could not book {slot_day.capitalize()} {slot_start}-{slot_end} (slot may be taken) — searching for another slot.")

            elif confirm == "NO":
                new_pref = input("Describe an alternative time(or Enter to retry): ").strip()
                if new_pref:
                    local_prefs = new_pref

            elif confirm == "SKIP":
                skipped.append(tool)
                break

            elif confirm == "CANCEL":
                terminated = True
                return bookings_log, skipped, terminated

    return bookings_log, skipped, terminated

# ═════════════════════════════════════════════════════════════════════════════
# USER SESSION
# ═════════════════════════════════════════════════════════════════════════════

async def run_user_session(username: str) -> None:    # Generates plan, perfroms booking and then logs everything

    print(f"\n Welcome, {username} ===\n")                    
    user_request = input("What would you like to make? ").strip()


    print("\n--- Generating plan ---") 
    approved_plan, steps, total_cost = await planning_phase(user_request)
    print(f"\n Plan approved. Total cost: €{total_cost}")

    needs_booking = [s for s in steps if s.get('requires_booking')]
    if needs_booking:
        print(f"\nSteps requiring booking ({len(needs_booking)}):")
        for s in needs_booking:
            print(f"  {s['step']}: {s['tool']} — {s['duration_hours']}h")

    preferences_text = discussion_phase(approved_plan, steps)


    if needs_booking:
        bookings_log, skipped, terminated = await booking_phase(
            steps, preferences_text, username
        )

    else:
        print("\nNo bookings required — all tools are freely available.")
        bookings_log, skipped, terminated = [], [], False




    log_path = write_session_log(
        username=username,
        user_request=user_request,
        approved_plan=approved_plan,
        bookings=bookings_log,
        total_cost=total_cost,
        skipped=skipped,
    )
    print(f"\n=== Session complete. Log: {log_path} ===")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    user = get_user()
    if user == "admin":
        await run_admin_session()
    else:
        await run_user_session(user)



if __name__ == "__main__":
    asyncio.run(main())