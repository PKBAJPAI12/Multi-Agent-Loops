import ast
import json
import logging
import operator
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_ITERATIONS = 20

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Understand the tool schema:
# Field           Purpose
# name            Identifies the tool the model can request
# description     Helps the model decide when to use it
# input_schema    JSON Schema describing valid arguments
# properties      Defines each argument and its data type
# required        Lists mandatory arguments

TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression, such as "
            "'1500 * 3' or '(100 + 50) / 2'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search a mock collection of web results. "
            "Use this to find information or product prices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The information to search for.",
                }
            },
            "required": ["query"],
        },
    },
]

GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters_json_schema=tool["input_schema"],
            )
            for tool in TOOLS
        ]
    )
]

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant with access to a calculator "
    "and a mock web search tool. Use tools when needed. "
    "For product prices, search first, then calculate if "
    "the user requests a total. Never describe mock results "
    "as verified live information."
)

# Only allow these arithmetic operations.
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression: str):
    """Evaluate a restricted arithmetic expression."""

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)

            # Prevent unnecessarily expensive exponentiation.
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            result = OPERATORS[type(node.op)](left, right)

            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported mathematical expression.")

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


def calculator(expression: str) -> str:
    try:
        result = safe_calculate(expression)
        return str(result)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculator error: {exc}"


def web_search(query: str) -> str:
    """Return mock search results; this is not a live web search."""

    mock_results = [
        {
            "title": "Laptop X",
            "price_inr": 50000,
            "description": "Example laptop listing.",
        },
        {
            "title": "Laptop Y",
            "price_inr": 65000,
            "description": "Example laptop listing.",
        },
        {
            "title": "Wireless Mouse",
            "price_inr": 1500,
            "description": "Example mouse listing.",
        },
    ]

    query_lower = query.lower()

    matches = [
        item
        for item in mock_results
        if any(
            word in (item["title"] + " " + item["description"]).lower()
            for word in query_lower.split()
            if len(word) > 2
        )
    ]

    if not matches:
        return json.dumps(
            {
                "query": query,
                "results": [],
                "note": "No matching items in the mock dataset.",
            }
        )

    return json.dumps(
        {
            "query": query,
            "results": matches,
            "note": "Mock data only; prices are illustrative.",
        }
    )


TOOL_HANDLERS = {
    "calculator": lambda tool_input: calculator(tool_input["expression"]),
    "web_search": lambda tool_input: web_search(tool_input["query"]),
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Execute the requested tool and return its result."""

    logger.info("Executing tool: %s | Input: %s", name, tool_input)

    try:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return f"Error: Unknown tool '{name}'."

        return handler(tool_input)

    except (KeyError, TypeError, ValueError) as exc:
        return f"Tool execution error: {exc}"


def run_agent(user_prompt: str) -> str:
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=GEMINI_TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        max_output_tokens=1024,
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info("Agent iteration %d/%d", iteration, MAX_ITERATIONS)

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        logger.info("Response: %s", response)
        function_calls = response.function_calls or []

        # Case 1: the model wants our application to execute one or more tools.
        if function_calls:
            contents.append(response.candidates[0].content)

            tool_response_parts = []
            for function_call in function_calls:
                logger.info("Function call: %s", function_call)
                result = execute_tool(
                    function_call.name,
                    dict(function_call.args or {}),
                )
                logger.info("Result: %s", result)
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": result},
                    )
                )
                logger.info("Tool response parts: %s", tool_response_parts)

            contents.append(
                types.Content(role="user", parts=tool_response_parts)
            )
            logger.info("Contents: %s", contents)
            continue

        # Case 2: the model has finished the task.
        if response.text:
            return response.text

        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason

        raise RuntimeError(
            f"Model stopped unexpectedly: {finish_reason}"
        )

    logger.warning(
        "Safety iteration cap reached: %d iterations",
        MAX_ITERATIONS,
    )

    raise RuntimeError(
        "Agent stopped because the safety iteration limit was reached."
    )


if __name__ == "__main__":
    prompt = (
        "Search for the price of a Wireless Mouse, "
        "then calculate the cost of buying 4 units."
    )

    try:
        answer = run_agent(prompt)

        print("\nFinal answer:")
        print(answer)

    except errors.APIError as exc:
        logger.exception("Gemini API request failed: %s", exc)

    except RuntimeError as exc:
        logger.error("Agent failed: %s", exc)
