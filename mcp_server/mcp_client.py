# file: mcp_client.py
# requirements for HTTP parts: requests
import subprocess, threading, json, requests, sys, time
from typing import Optional, Iterator


class StdioTransport:
    def __init__(self, cmd):
        # cmd can be a list or string; we use subprocess
        if isinstance(cmd, str):
            cmd = cmd.split()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        # start a thread to read stdout lines into a queue or callback if needed
        self._recv_lock = threading.Lock()

    def send(self, obj):
        msg = json.dumps(obj, ensure_ascii=False) + "\n"
        self.proc.stdin.write(msg)
        self.proc.stdin.flush()

    def recv_one(self, timeout: Optional[float] = None):
        # blocking read one line (simplified)
        line = self.proc.stdout.readline()
        if not line:
            return None
        return json.loads(line)

class HTTPTransport:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def send(self, obj):
        r = requests.post(f"{self.base_url}/mcp", json=obj, timeout=30)
        r.raise_for_status()
        return r.json()

    def stream(self, obj) -> Iterator[dict]:
        # call /mcp/stream
        r = requests.post(f"{self.base_url}/mcp/stream", json=obj, stream=True, timeout=60)
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                yield {"_raw": line}

class MCPClient:
    def __init__(self, url: str):
        self.url = url
        if url.startswith("stdio://"):
            cmd = url[len("stdio://"):]
            self.transport = StdioTransport(cmd)
            self.mode = "stdio"
        elif url.startswith("http://") or url.startswith("https://"):
            self.transport = HTTPTransport(url)
            self.mode = "http"
        else:
            raise ValueError("Unsupported scheme")

        self._next_id = 1

    def _next(self):
        i = self._next_id
        self._next_id += 1
        return i

    def initialize(self, capabilities=None):
        req = {"jsonrpc":"2.0",
               "id": self._next(),
               "method":"initialize",
               "params":{"protocolVersion":"2025-06-18","capabilities":capabilities or {}}
               }
        if self.mode == "stdio":
            self.transport.send(req)
            return self.transport.recv_one()
        else:
            return self.transport.send(req)

    def list_tools(self):
        req = {"jsonrpc":"2.0",
               "id": self._next(),
               "method":"tools/list"}
        if self.mode == "stdio":
            self.transport.send(req)
            return self.transport.recv_one()
        else:
            return self.transport.send(req)

    def call_tool(self, name, arguments=None, stream=False):
        req = {"jsonrpc":"2.0","id": self._next(),"method":"tools/call","params":{"name":name,"arguments":arguments or {}}}
        if self.mode == "stdio":
            self.transport.send(req)
            return self.transport.recv_one()
        else:
            if stream:
                # iterate chunks
                for chunk in self.transport.stream(req):
                    yield chunk
            else:
                return self.transport.send(req)

# usage example
if __name__ == "__main__":
    # example1: stdio server exe path (start the stdio server we wrote earlier)
    # client = MCPClient("stdio://python mcp_stdio_server.py")  # 注意：在shell里要能执行
    # example2: http server
    client = MCPClient("http://localhost:8080")

    print("initialize ->", client.initialize({"elicitation": {}}))
    tl = client.list_tools()
    print("tools/list ->", tl)

    # call read_file (non-stream)
    resp = client.call_tool("read_file", {"path":"./data/demo.txt"})
    print("tools/call read_file ->", resp)

    # call long_process streaming
    for idx, chunk in enumerate(client.call_tool("long_process", {"whatever": True}, stream=True)):
        print("stream chunk:", chunk)
