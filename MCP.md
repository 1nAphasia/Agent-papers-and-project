## 简介

MCP 是 Model Context Protocol 的缩写，Model 强调服务主体是 LLM，Context 强调其**信息枢纽功能**，Protocol 则**凸显信息交互的标准化特性**，MCP 如同 USB-C 接口般**通过统一协议实现 LLM 与外部能力的高效互联**。

MCP（模型上下文协议）是一种开源标准，用于将AI应用程序连接到外部系统。通过MCP，像Claude或ChatGPT这样的AI应用可以**连接数据源（如本地文件、数据库）、工具（如搜索引擎、计算器）和工作流（如专用提示词）**，从而获取关键信息并执行任务。

### MCP规定了什么？

MCP规定的核心内容：

- 协议栈：两层设计
  - 数据层：设计为基于 JSON-RPC 2.0 的消息/方法/通知，定义了会话生命周期、能力协商、核心原语（tools/resources/prompts）、通知与长任务等。这是MCP Client与MCP Server通信的数据格式。
  - 传输层：定义把 JSON-RPC 消息在进程间或网络上传输的方式。当前支持 stdio 和 Streamable HTTP/SSE 等，包括分帧与认证建议。
- 会话与生命周期：规范 initialize 握手（协议版本、capabilities 协商、clientInfo/serverInfo）、就绪通知、终止流程等。
- 三个服务器端的基本要素：
  - Tools — 可执行的函数/动作（发现：tools/list，执行：tools/call）。
  - Resources — 可读取的上下文数据（例如文件、数据库记录、索引片段），有 resources/list / resources/read 等接口。
  - Prompts — 复用的提示模板或 few-shot 示例，可供 client/agent 使用。
- 客户端暴露能力：使 MCP server 可以请求客户端进行模型采样、进行用户提示、或记录日志。

总而言之,MCP通过协议约定一系列内容,使得接口标准化。进而可以通过完成接口的适配来使得一般的LLM可以广泛的调用各种工具,包括本地提供的工具和数据以及浩如烟海的互联网各路服务商提供的工具与数据(如地图、天气数据等)。

### MCP Server

MCP server在实现上与一般的网络服务器没有太大差别,只是遵循MCP协议暴露其能力、通信标准和方式。本地服务器通过stdio通信、非本地服务器则通过HTTP、SSE方式通信。

MCP SDK提供了一个简单FastMCP实现,允许Client通过单个脚本(py、node)来启动本地服务器进行互联。也可拓展之使用httpx连接,获取互联网其他提供商提供的MCP Server的服务。

```python

[{
  'type': 'function', 
  'function': {
  'name': 'get_alerts', 
  'description': 'Get weather alerts for a US state.', 
  'parameters': {'properties': {'state': {'title': 'State', 'type': 'string'}}, 
  'required': ['state'], 
  'title': 'get_alertsArguments', 'type': 'object'}}
}]

# 可以对应如下函数

@mcp.tool(description="Get weather alerts for a US state.")
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

```

### 在项目中使用

