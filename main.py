from anthropic import Anthropic
import os
import subprocess
from dotenv import load_dotenv
import json
from pathlib import Path
import glob

WORKDIR = Path.cwd()

white = "\033[97m"
yellow = "\033[33m"
reset = "\033[0m"

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
    },
    {
        "name": "write",
        "description": "write text into a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "text": {
                    "type": "string"
                }
            },
            "required": ["path", "text"]
        }
    },
    {
        "name": "read",
        "description": "read content from a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "edit",
        "description": "repalce exact file content from old_text to new_text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "old_text": {
                    "type": "string"
                },
                "new_text": {
                    "type": "string"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
    {
        "name": "find_paths",
        "description": "Find files and directories under a given directory that match a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
                "pattern": {
                    "type": "string",
                }
            },
            "required": ["path", "pattern"]
        }
    }
]

client = Anthropic()

DENY_LIST = ("rm -rf /", "shutdown", "reboot")
        
def run_read(path: Path) -> str:
    path = (WORKDIR / path).resolve()
    if not path.is_relative_to(WORKDIR):
        return f"{path} is out of workspace"
    output = path.read_text(encoding="utf-8")
    return output

def run_write(path: Path, text: str) -> str:
    path = (WORKDIR / path).resolve()
    if not path.is_relative_to(WORKDIR):
        return f"{path} is out of workspace"
    output = path.write_text(text, encoding="utf-8")
    return f"wrote {output} characters to {path}"

def run_edit(path: Path, old_text: str, new_text) -> str:
    path = (WORKDIR / path).resolve()
    if not path.is_relative_to(WORKDIR):
        return f"{path} is out of space"
    text = path.read_text(encoding="utf-8")
    if old_text not in text:
        return f"{old_text} not found"
    path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"edited {path}"

def run_glob(path: Path, pattern: str) -> str:
    path = (WORKDIR / path).resolve()
    if not path.is_relative_to(WORKDIR):
        return f"{path} is out of workspace"
    matches =  (match for match in glob.glob(
        pattern, root_dir=path, recursive=True
    ) if (path / match).resolve().is_relative_to(WORKDIR))
    return "\n".join(matches) or "no match"
    

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


TOOL_HANDLERS = {
    "shell": run_shell,
    "read": run_read,
    "write": run_write,
    "edit": run_edit,
    "find_paths": run_glob
}

def execute_tools(tool_name: str, args: dict) -> str:
    tool = TOOL_HANDLERS[tool_name]

    if tool_name == "shell":
        print(f"{white}$ {args["command"]}{reset}")
    else:
        print(f"{white}{tool_name} {yellow}{args["path"]}{reset}")
    output = tool(**args)
    print(f"\033[090m{output}\033[0m\n")
    return output


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
            output = execute_tools(block.name, block.input)

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
