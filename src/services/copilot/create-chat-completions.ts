import consola from "consola"
import { events } from "fetch-event-stream"

import { copilotHeaders, copilotBaseUrl } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { state } from "~/lib/state"

export const createChatCompletions = async (
  payload: ChatCompletionsPayload,
) => {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  const enableVision = payload.messages.some(
    (x) =>
      typeof x.content !== "string"
      && x.content?.some((x) => x.type === "image_url"),
  )

  // Agent/user check for X-Initiator header
  // Determine if any message is from an agent ("assistant" or "tool")
  const isAgentCall = payload.messages.some((msg) =>
    ["assistant", "tool"].includes(msg.role),
  )

  // Build headers and add X-Initiator
  const headers: Record<string, string> = {
    ...copilotHeaders(state, enableVision),
    "X-Initiator": isAgentCall ? "agent" : "user",
  }

  const initiator = isAgentCall ? "agent" : "user"
  const reasoning = payload.reasoning_effort ?? "none"
  const stream = payload.stream ? "stream" : "sync"
  consola.info(
    `<-- ${payload.model} | initiator=${initiator} | reasoning=${reasoning} | vision=${enableVision ? "yes" : "no"} | ${stream}`,
  )

  const start = Date.now()
  const response = await fetch(`${copilotBaseUrl(state)}/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    consola.error("Failed to create chat completions", response)
    throw new HTTPError("Failed to create chat completions", response)
  }

  if (payload.stream) {
    return events(response)
  }

  const result = (await response.json()) as ChatCompletionResponse
  for (const choice of result.choices) {
    normalizeReasoningFields(choice.message)
  }
  const elapsed = ((Date.now() - start) / 1000).toFixed(1)
  const usage = result.usage
  consola.info(
    `--> ${response.status} ${elapsed}s | tokens: in=${usage?.prompt_tokens ?? 0} out=${usage?.completion_tokens ?? 0}`,
  )

  return result
}

// Streaming types

export interface ChatCompletionChunk {
  id: string
  object: "chat.completion.chunk"
  created: number
  model: string
  choices: Array<Choice>
  system_fingerprint?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_tokens_details?: {
      cached_tokens: number
    }
    completion_tokens_details?: {
      accepted_prediction_tokens: number
      rejected_prediction_tokens: number
    }
  }
}

interface Delta {
  content?: string | null
  reasoning_content?: string | null
  role?: "user" | "assistant" | "system" | "tool"
  tool_calls?: Array<{
    index: number
    id?: string
    type?: "function"
    function?: {
      name?: string
      arguments?: string
    }
  }>
}

interface Choice {
  index: number
  delta: Delta
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null
  logprobs: object | null
}

// Non-streaming types

export interface ChatCompletionResponse {
  id: string
  object: "chat.completion"
  created: number
  model: string
  choices: Array<ChoiceNonStreaming>
  system_fingerprint?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_tokens_details?: {
      cached_tokens: number
    }
  }
}

interface ResponseMessage {
  role: "assistant"
  content: string | null
  reasoning_content?: string | null
  tool_calls?: Array<ToolCall>
}

interface ChoiceNonStreaming {
  index: number
  message: ResponseMessage
  logprobs: object | null
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter"
}

// Payload types

export interface ChatCompletionsPayload {
  messages: Array<Message>
  model: string
  temperature?: number | null
  top_p?: number | null
  max_tokens?: number | null
  stop?: string | Array<string> | null
  n?: number | null
  stream?: boolean | null

  frequency_penalty?: number | null
  presence_penalty?: number | null
  logit_bias?: Record<string, number> | null
  logprobs?: boolean | null
  response_format?: { type: "json_object" } | null
  seed?: number | null
  tools?: Array<Tool> | null
  tool_choice?:
    | "none"
    | "auto"
    | "required"
    | { type: "function"; function: { name: string } }
    | null
  user?: string | null
  reasoning_effort?: "low" | "medium" | "high" | null
}

export interface Tool {
  type: "function"
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown>
  }
}

export interface Message {
  role: "user" | "assistant" | "system" | "tool" | "developer"
  content: string | Array<ContentPart> | null

  name?: string
  tool_calls?: Array<ToolCall>
  tool_call_id?: string
}

export interface ToolCall {
  id: string
  type: "function"
  function: {
    name: string
    arguments: string
  }
}

export type ContentPart = TextPart | ImagePart

export interface TextPart {
  type: "text"
  text: string
}

export interface ImagePart {
  type: "image_url"
  image_url: {
    url: string
    detail?: "low" | "high" | "auto"
  }
}

/**
 * Copilot gateway returns `reasoning_text` instead of the OpenAI standard
 * `reasoning_content`. Normalize to `reasoning_content` for compatibility.
 */
export function normalizeReasoningFields(obj: object): void {
  const raw = obj as Record<string, unknown>
  if ("reasoning_text" in raw) {
    raw.reasoning_content = raw.reasoning_text
    delete raw.reasoning_text
    delete raw.reasoning_opaque
  }
}
