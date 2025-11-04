from mcp.server.fastmcp import FastMCP
from typing import Optional

mcp=FastMCP("demo")

@mcp.tool()

def add(a:int,b:int)-> int:
    return a+b


@mcp.resource("greeting://{name}")
def get_greeting(name:str)->str:
    return f"hello:{name}!"

@mcp.prompt()
def greet_user(name:str,style:str="friendly")->str:
    styles={
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }
    return f"{styles.get(style, styles['friendly'])} for someone named {name}."

def main():
    # Initialize and run the server
    mcp.run(transport='stdio')    
    print(mcp.list_tools())

if __name__ == "__main__":
    main()