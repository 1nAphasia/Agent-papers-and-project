import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config.logger import get_logger

logger=get_logger(__name__)

class MCPClient:
    def __init__(self):

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.server_tools=[]

    async def connect_to_server(self,server_script_path:str):

      is_python=server_script_path.endswith('.py')
      is_js=server_script_path.endswith('.js')
      if not (is_python or is_js):
          raise ValueError("Server script must be a .py or .js file")
      
      command="python" if is_python else "node"
      server_params=StdioServerParameters(
          command=command,
          args=[server_script_path],
          env=None
      )

      stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
      self.stdio, self.write = stdio_transport
      self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
      await self.session.initialize()
        
      response=await self.session.list_tools()
              
      self.server_tools=response.tools

      print("\nConnected to server with tools:",[tool.name for tool in self.server_tools])

    def list_tools(self):
        '''
        一个返回的schema的实例：
        [
          {
            "name": "add_note",
            "description": "Add a new note to database",
            "inputSchema": {
              "type": "object",
              "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"}
              },zz
              "required": ["title", "content"]
            }
          }
        ]
        '''
        logger.info("当前连接的服务器所拥有的工具："+str(self.server_tools))
        return self.server_tools
    async def call_tools(self,tool_name,tool_args):
        results=await self.session.call_tool(tool_name,tool_args)
        logger.info(f"使用参数{str(tool_args)} 调用工具 {tool_name} \n 返回结果 {str(results)}")
        return results
        

    async def shutdown(self):
        await self.exit_stack.aclose()


