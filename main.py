from anthropic import Anthropic
import os
import subprocess
from dotenv import load_dotenv
import json

load_dotenv(override=True)

TOOLS = [
    {
        "name": "shell",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]

client = Anthropic()

DENY_LIST = ("rm -rf /", "shutdown", "reboot")


def write_api_log(record: dict) -> None:
    line = json.dumps(
        record, ensure_ascii=False, default=lambda obj: obj.model_dump(mode="json")
    )
    with open("api.jsonl", "a", encoding="utf-8") as log:
        log.write(line + "\n")


def run_shell(command: str) -> str:
    for deny in DENY_LIST:
        if deny in command:
            return f"Blocked: {deny} is dangerous, we can not execute it."
    try:
        res = subprocess.run(
            command,
            shell=True,
            errors="replace",
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return f"Command time out after 120s"
    except OSError:
        return f"Could not execute command: {command}"
    
    output = (res.stdout + res.stderr).strip()
    return output[:50000] if output else "(no output)"


def agent_loop(messages: list):
    while True:
        write_api_log(
            {
                "request": {
                    "model": os.getenv("MODEL_ID"),
                    "max_tokens": 4096,
                    "messages": messages,
                    "tools": TOOLS,
                }
            }
        )
        response = client.messages.create(
            messages=messages, model=os.getenv("MODEL_ID"), max_tokens=4096, tools=TOOLS
        )
        write_api_log({"response": response})
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_call = [block for block in response.content if block.type == "tool_use"]
        if not tool_call:
            break

        tool_results = []

        for block in tool_call:
            command = block.input["command"]
            print(f"\033[033m{command}\033[0m")
            output = run_shell(command)
            print(f"\033[090m{output}\033[0m")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[036m> \033[0m").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print() #为了不让下一条终端提示符接在 CTRL + C 后面
            break
        if not query:
            continue
        if query.lower() in ("quit", "q", "exit"):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        ans = history[-1]["content"]
        for block in (block for block in ans if block.type == "text"):
            print(block.text)
