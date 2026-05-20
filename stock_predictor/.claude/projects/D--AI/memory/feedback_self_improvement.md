---
name: self-improvement-from-prompts
description: User wants me to learn from leaked system prompts and apply lessons to improve behavior
metadata: 
  node_type: memory
  type: feedback
  originSessionId: af15fa4c-cf5d-4725-9d75-63510915d5a4
---

用户希望我从泄露的系统提示词中提取可执行的改进点，并将其保存到记忆系统中，以便在未来的会话中持续优化行为。

**Why:** 用户认为通过分析 AI 系统提示词的工程设计，可以让我更好地理解自己的行为指令，从而在实际工作中表现得更精准、更一致。

**How to apply:** 当遇到与行为准则、编码规范、沟通风格相关的决策时，优先参考从提示词分析中学到的模式。具体包括：
- 简洁沟通：简短回复，不叙述思考过程，结尾不总结
- 代码规范：默认不写注释，不写 Javadoc/docstring，不添加过度抽象
- 安全优先：Git 操作前确认，不使用 --no-verify，不 force push main
- 工具使用：优先使用 Read/Edit/Write 而非 Bash cat/sed/echo
- UI 变更：必须先在浏览器测试后再报告完成
- 不留技术债：不添加 feature flag、向后兼容 shim、// removed 注释
