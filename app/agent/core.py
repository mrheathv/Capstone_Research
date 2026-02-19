import json
import streamlit as st
from openai import OpenAI
from typing import Dict, Any
from .tools import TOOLS, get_tools_for_openai

def get_openai_client():
    """Initialize and return an OpenAI client"""
    return OpenAI()

def agent_answer(user_question: str, max_iterations: int = 5, current_user: str = None) -> str:
    """
    Agent that uses ReAct pattern to answer questions with multiple tools.

    Args:
        user_question: The user's natural language question
        max_iterations: Maximum number of reasoning loops (safety limit)
        current_user: Optional override for the current sales agent name.
                      Falls back to st.session_state['current_user'] if not provided.

    Returns:
        Final synthesized answer as a string
    """
    client = get_openai_client()

    # Convert tools to OpenAI format
    tools_for_openai = get_tools_for_openai()

    if current_user is None:
        current_user = st.session_state.get('current_user', 'Unknown')

    system_message = f"""You are a helpful sales assistant with access to a CRM database.

    Current User: {current_user}

    You have three tools available:
    - recommend_contacts: Use this when the user asks who to contact, wants contact recommendations,
      or has free time and wants to know who to reach out to. This tool scores accounts by
      propensity to buy, revenue, and days since last contact.
    - text_to_sql: For flexible, ad-hoc queries about any data in the database. Use this when
      the user asks for summaries, history, deal details, or any specific data lookup.
    - open_work: For quickly getting outstanding work items, follow-ups, meetings, or open
      engagements. Automatically filtered for the current user.

    ROUTING RULES — follow these precisely:
    1. "Availability / who to contact / recommendations" → call recommend_contacts first.
    2. "Summary of interactions with [account name]" → call text_to_sql with a query that
       retrieves interaction history for that specific account.
    3. "Follow-up / open tasks / meetings / open engagements" → call open_work first.
    4. For multi-part questions: gather all needed information, then synthesize a single answer.

    GUARDRAILS:
    - Only read data — never suggest or attempt to write, update, or delete records.
    - If a question is outside the scope of the CRM database, politely say so.

    Do NOT just return raw tool output — always provide a clear, concise, synthesized answer
    after gathering information. Format the answer in plain markdown for readability.
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_question}  
    ]

    try: 
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}")
            print(f"{'='*60}")
            # Ask LLM what to do next
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                tools=tools_for_openai,
                tool_choice='auto'
            )

            message = response.choices[0].message

            # if no tool calls, LLM has final answer
            if not message.tool_calls:
                print("\n✓ LLM PROVIDED FINAL ANSWER (no more tool calls)")
                print(f"Answer: {message.content[:200]}...")
                return message.content or "I'm not sure how to help with that."
                   

            print(f"\n→ LLM WANTS TO CALL {len(message.tool_calls)} TOOL(S):")
            for tc in message.tool_calls:
                print(f"  - {tc.function.name}({tc.function.arguments})")
            # Assistant's reasoning to conversation
            # Append the assistant's message so that the LLM remembers wht it just decided to do. Without it
            # the conversation would have gaps. 
            messages.append(message)
            print(f"\n→ ADDED assistant message to conversation (now {len(messages)} messages)")

            # Execute each tool the LLM requested
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"\n→ EXECUTING: {tool_name}")

                tool = TOOLS.get(tool_name)
                if not tool:
                    result = f"Error: Tool '{tool_name}' not found."
                else:
                    result = tool.handler(tool_args)

                print(f"→ RESULT: {len(result)} characters")
                print(f"  Preview: {result[:150]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result
                })
                print(f"→ ADDED tool result to conversation (now {len(messages)} messages)")
            
            # End of inner for loop - back to outer iteration loop
            print(f"\n→ END OF ITERATION {iteration + 1}")
        
        # OUTSIDE the for loop (dedent twice - align with 'for iteration')
        return "I've gathered information but reached my processing limit"
        
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"
    
