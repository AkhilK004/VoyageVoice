"""
tools.py — Tool definitions for the VoyageVoice travel agent.

Tools are defined as OpenAI/Groq function-calling schemas.
Groq supports function calling natively with Llama 3 models.

Available tools:
  - check_visa_eligibility(passport_country, destination)
  - get_required_documents(passport_country, destination)
  - get_processing_time(passport_country, destination)
  - escalate_to_human(reason)
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Load visa data once
_DATA_PATH = Path(__file__).parent.parent / "knowledge" / "data" / "visa_data.json"
_VISA_DATA: list = []


def _get_visa_data() -> list:
    global _VISA_DATA
    if not _VISA_DATA:
        _VISA_DATA = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return _VISA_DATA


def _find_visa_entry(passport_country: str, destination: str) -> dict | None:
    """Find the best-matching visa type for the given passport/destination combo."""
    data = _get_visa_data()
    destination_clean = destination.lower().strip()

    for entry in data:
        if destination_clean in entry["country"].lower():
            # Find the visa type that covers this passport
            for vt in entry["visa_types"]:
                eligible = [p.lower() for p in vt["eligible_passports"]]
                if passport_country.lower() in eligible:
                    return {"country": entry["country"], "visa_type": vt}

            # If no specific match, return the first visa type with full data
            if entry["visa_types"]:
                return {"country": entry["country"], "visa_type": entry["visa_types"][0]}

    return None


# ── Tool Implementations ──────────────────────────────────────────────────────

def check_visa_eligibility(passport_country: str, destination: str) -> dict:
    """
    Check if a passport holder needs a visa and what type.
    Returns eligibility info with visa type name and key facts.
    """
    result = _find_visa_entry(passport_country, destination)
    if not result:
        return {
            "found": False,
            "message": f"No visa information found for {passport_country} passport holders traveling to {destination}. Please contact the nearest embassy."
        }

    vt = result["visa_type"]
    eligible = [p.lower() for p in vt["eligible_passports"]]
    is_eligible = passport_country.lower() in eligible

    return {
        "found": True,
        "country": result["country"],
        "passport_country": passport_country,
        "visa_type": vt["type"],
        "visa_on_arrival": vt["visa_on_arrival"],
        "e_visa_available": vt["e_visa"],
        "duration": vt["duration"],
        "fees_usd": vt["fees_usd"],
        "fees_inr": vt["fees_inr"],
        "notes": vt["notes"],
    }


def get_required_documents(passport_country: str, destination: str) -> dict:
    """
    Return the list of required documents for a visa application.
    """
    result = _find_visa_entry(passport_country, destination)
    if not result:
        return {
            "found": False,
            "message": f"No document information found for {passport_country} to {destination}."
        }

    vt = result["visa_type"]
    return {
        "found": True,
        "country": result["country"],
        "visa_type": vt["type"],
        "documents": vt["required_documents"],
        "document_count": len(vt["required_documents"]),
    }


def get_processing_time(passport_country: str, destination: str) -> dict:
    """
    Return estimated visa processing time and fees.
    """
    result = _find_visa_entry(passport_country, destination)
    if not result:
        return {
            "found": False,
            "message": f"No processing time data found for {passport_country} to {destination}."
        }

    vt = result["visa_type"]
    return {
        "found": True,
        "country": result["country"],
        "visa_type": vt["type"],
        "processing_time": vt["processing_time_days"],
        "fees_usd": vt["fees_usd"],
        "fees_inr": vt["fees_inr"],
        "validity": vt["validity"],
    }


def escalate_to_human(reason: str) -> dict:
    """
    Signal that this conversation should be escalated to a human agent.
    """
    logger.info(f"Escalation triggered: {reason}")
    return {
        "escalated": True,
        "reason": reason,
        "message": "I'm connecting you with one of our visa specialists now. They'll be able to help you further.",
    }


# ── Tool Schemas (for Groq function calling) ──────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_visa_eligibility",
            "description": "Check whether a traveller needs a visa, what type of visa, and key details like fees and duration. Use this when the user asks 'do I need a visa' or 'can I travel to X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passport_country": {
                        "type": "string",
                        "description": "The traveller's passport country (e.g., 'India', 'USA', 'UK')"
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination country they want to visit (e.g., 'Japan', 'Thailand', 'USA')"
                    }
                },
                "required": ["passport_country", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_required_documents",
            "description": "Get the list of required documents for a visa application. Use when user asks 'what documents do I need'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passport_country": {
                        "type": "string",
                        "description": "The traveller's passport country"
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination country"
                    }
                },
                "required": ["passport_country", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_processing_time",
            "description": "Get visa processing time and fees. Use when user asks about how long it takes or how much it costs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passport_country": {
                        "type": "string",
                        "description": "The traveller's passport country"
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination country"
                    }
                },
                "required": ["passport_country", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the conversation to a human visa specialist. Use when the query is complex, involves an existing application, or the traveller is frustrated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for escalation (e.g., 'complex visa situation', 'existing application issue')"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]


# ── Tool Dispatcher ───────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "check_visa_eligibility": check_visa_eligibility,
    "get_required_documents": get_required_documents,
    "get_processing_time": get_processing_time,
    "escalate_to_human": escalate_to_human,
}


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool by name with the given arguments."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = TOOL_REGISTRY[tool_name](**arguments)
        logger.info(f"Tool '{tool_name}' called with {arguments} → {result}")
        return result
    except Exception as e:
        logger.error(f"Tool '{tool_name}' error: {e}")
        return {"error": str(e)}
