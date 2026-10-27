import { resolveProviderApiKey } from '../../../shared/provider-api-key.mjs';
import { getLlmDispatcher } from '../../../shared/llm/http-agent.mjs';
import { sanitizeToolPairs } from '../session/context-utils.mjs';
import { buildGeminiRequest } from './gemini-request-body.mjs';
import { geminiBareModelId } from './gemini-cache-policy.mjs';
import { geminiEndsWithFunctionResponse, runGeminiStreamAttempts } from './gemini-retry-policy.mjs';
import { consumeGeminiRestStreamResponse } from './gemini-stream.mjs';
import { geminiTextLeakGuardFor } from './gemini-sdk-request.mjs';
import { createGeminiReasoningRelay } from './gemini-reasoning-relay.mjs';
import {
  geminiFailureUsage,
  geminiIncompleteError,
  geminiSendResult,
  parseGeminiCandidate,
  resolveGeminiUsage,
} from './gemini-response.mjs';
import { emitGeminiToolCalls } from './gemini-schema.mjs';
import { geminiRestError } from './gemini.mjs';
import { canFallbackNonStreaming } from './retry-classifier.mjs';
import { buildAnthropicApiRequest } from './anthropic-api-request.mjs';
import { applyAnthropicErrorContract } from './anthropic-error-contract.mjs';
import { createAnthropicMidState, assertAnthropicStreamNotEmpty } from './anthropic-midstream-recovery.mjs';
import { parseSSEStream } from './anthropic-sse.mjs';
import { buildAnthropicTurnResult } from './anthropic-turn-result.mjs';
import { streamCallbacks } from './lib/send-callbacks.mjs';
import { preparePdfTextForProvider } from './media-normalization.mjs';
import { GEMINI_MODELS } from './lib/gemini-model-catalog.mjs';
import { MODELS as ANTHROPIC_MODELS } from './lib/anthropic-models.mjs';
import { VertexAccessTokenManager } from './vertex-ai-auth.mjs';

const VERTEX_ANTHROPIC_VERSION = 'vertex-2023-10-16';

function vertexModel(model) {
  const raw = String(model || '').trim();
  if (!raw) throw new Error('[vertex-ai] model is required');
  const publisherPath = /^publishers\/(google|anthropic)\/models\/(.+)$/.exec(raw);
  if (publisherPath) return { publisher: publisherPath[1], id: publisherPath[2] };
  const shortPath = /^(google|anthropic)\/(.+)$/.exec(raw);
  if (shortPath) return { publisher: shortPath[1], id: shortPath[2] };
  if (/^gemini-/i.test(raw)) return { publisher: 'google', id: geminiBareModelId(raw) };
  if (/^claude-/i.test(raw)) return { publisher: 'anthropic', id: raw };
  throw new Error(`[vertex-ai] cannot determine publisher for model "${raw}"`);
}

function vertexHost(location) {
  return location === 'global' ? 'aiplatform.googleapis.com' : `${location}-aiplatform.googleapis.com`;
}

function vertexLocation(config) {
  const location = String(config?.location || 'global').trim().toLowerCase();
  if (!/^[a-z0-9-]{1,63}$/.test(location)) throw new Error('[vertex-ai] location is invalid');
  return location;
}

function vertexProjectOverride(config) {
  const projectId = String(config?.projectId || '').trim();
  if (projectId && (!/^[a-zA-Z0-9._:-]{1,128}$/.test(projectId) || projectId.includes('/'))) {
    throw new Error('[vertex-ai] project ID override is invalid');
  }
  return projectId;
}

function geminiCallbacks(opts) {
  const reasoning = createGeminiReasoningRelay(opts);
  return {
    onStreamDelta: typeof opts.onStreamDelta === 'function' ? opts.onStreamDelta : null,
    onToolCall: typeof opts.onToolCall === 'function' ? opts.onToolCall : null,
    onTextDelta: typeof opts.onTextDelta === 'function' ? opts.onTextDelta : null,
    onReasoningDelta: reasoning.onReasoningDelta,
    retractReasoning: reasoning.retract,
  };
}

function geminiVertexBody(request) {
  return {
    contents: request.contents,
    ...(request.systemInstruction ? { systemInstruction: request.systemInstruction } : {}),
    ...(request.generationConfig ? { generationConfig: request.generationConfig } : {}),
    ...(request.geminiTools ? { tools: request.geminiTools } : {}),
    ...(request.toolConfig ? { toolConfig: request.toolConfig } : {}),
    ...(request.safetySettings ? { safetySettings: request.safetySettings } : {}),
  };
}

export class VertexAIProvider {
  static inputExcludesCache = true;
  nativePdf = true;
  name = 'vertex-ai';

  constructor(config = {}) {
    this.config = config;
    this._fetch = typeof config.fetchFn === 'function' ? config.fetchFn : fetch;
    this._auth = new VertexAccessTokenManager({
      credentialResolver: () => resolveProviderApiKey('vertex-ai', this.config),
      fetchFn: this._fetch,
      now: typeof config.now === 'function' ? config.now : () => Date.now(),
    });
  }

  _projectId() {
    return vertexProjectOverride(this.config) || this._auth.projectId();
  }

  _url(publisher, model, method) {
    const location = vertexLocation(this.config);
    const project = encodeURIComponent(this._projectId());
    const id = encodeURIComponent(model);
    return `https://${vertexHost(location)}/v1/projects/${project}/locations/${location}/publishers/${publisher}/models/${id}:${method}`;
  }

  async _authorizedFetch(url, init) {
    const run = async (force) => {
      const token = await this._auth.getAccessToken({ force });
      const headers = new Headers(init?.headers || {});
      headers.set('Authorization', `Bearer ${token}`);
      return await this._fetch(url, { ...init, headers, dispatcher: getLlmDispatcher() });
    };
    let response = await run(false);
    if (response.status === 401) {
      this._auth.invalidate();
      response = await run(true);
    }
    return response;
  }

  async send(messages, model, tools, sendOpts = {}) {
    const route = vertexModel(model);
    if (route.publisher === 'google') return await this._sendGemini(messages, route.id, tools, sendOpts);
    return await this._sendClaude(messages, route.id, tools, sendOpts);
  }

  async _sendGemini(messages, model, tools, opts) {
    const signal = opts.signal || null;
    const callbacks = geminiCallbacks(opts);
    const request = buildGeminiRequest(messages, model, tools, opts, null);
    const body = geminiVertexBody(request);
    const stream = { ...request, opts, signal, useModel: model, tools, callbacks, cachedContent: null };
    try {
      opts.onStageChange?.('requesting');
    } catch {}

    let outcome;
    try {
      outcome = await runGeminiStreamAttempts({
        signal,
        sleepFn: this.config.sleepFn,
        afterToolResult: geminiEndsWithFunctionResponse(request.contents),
        failureUsage: geminiFailureUsage(opts, null, model),
        run: async ({ signal: attemptSignal, connected }) => {
          callbacks.retractReasoning();
          const url = `${this._url('google', model, 'streamGenerateContent')}?alt=sse`;
          const res = await this._authorizedFetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: attemptSignal,
          });
          if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw geminiRestError(res, text, 'Vertex AI Gemini streamGenerateContent');
          }
          connected();
          try {
            opts.onStageChange?.('streaming');
          } catch {}
          const textLeakGuard = geminiTextLeakGuardFor(stream);
          const response = await consumeGeminiRestStreamResponse(res, {
            signal: attemptSignal,
            onStreamDelta: callbacks.onStreamDelta,
            onTextDelta: callbacks.onTextDelta,
            onReasoningDelta: callbacks.onReasoningDelta,
            textLeakGuard,
            label: 'Vertex AI Gemini streamGenerateContent',
            failureUsage: geminiFailureUsage(opts, null, model),
          });
          return { response, textLeakGuard };
        },
      });
    } catch (streamError) {
      if (!canFallbackNonStreaming(streamError, { signal })) throw streamError;
      callbacks.retractReasoning();
      const res = await this._authorizedFetch(this._url('google', model, 'generateContent'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw geminiRestError(res, text, 'Vertex AI Gemini generateContent');
      }
      outcome = { response: await res.json(), textLeakGuard: null };
    }

    const parsed = parseGeminiCandidate(outcome.response, outcome.textLeakGuard);
    const usage = resolveGeminiUsage(outcome.response, opts, null, model, this.name);
    const incomplete = geminiIncompleteError(outcome.response, parsed, model, usage);
    if (incomplete) throw incomplete;
    outcome.textLeakGuard?.releaseDeferredToolCalls?.();
    emitGeminiToolCalls(parsed.nativeToolCalls, callbacks.onToolCall);
    return geminiSendResult(parsed, model, opts, usage);
  }

  async _sendClaude(messages, model, tools, opts) {
    messages = sanitizeToolPairs(messages);
    messages = await preparePdfTextForProvider(messages, this);
    const requestOpts = { ...opts, fast: false, serverFallback: false };
    const { params, knownToolNames } = await buildAnthropicApiRequest({
      name: this.name,
      config: { ...this.config, disableBetaHeaders: true },
      messages,
      useModel: model,
      tools,
      opts: requestOpts,
      fastModeLatched: false,
      capabilityModel: model,
    });
    const { model: _model, ...body } = params;
    body.anthropic_version = VERTEX_ANTHROPIC_VERSION;
    const callbacks = streamCallbacks(opts);
    try {
      callbacks.onStageChange?.('requesting');
    } catch {}
    const response = await this._authorizedFetch(this._url('anthropic', model, 'streamRawPredict'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: opts.signal || null,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      const error = Object.assign(
        new Error(`Vertex AI Claude streamRawPredict ${response.status}: ${text.slice(0, 300)}`),
        {
          status: response.status,
          httpStatus: response.status,
          headers: response.headers,
          response: { status: response.status, headers: response.headers },
          initialResponseError: true,
        }
      );
      throw applyAnthropicErrorContract(error, { status: response.status, headers: response.headers, bodyText: text });
    }
    try {
      callbacks.onStageChange?.('streaming');
    } catch {}
    const controller = new AbortController();
    const externalSignal = opts.signal || null;
    const abort = () => controller.abort(externalSignal?.reason);
    if (externalSignal) {
      if (externalSignal.aborted) abort();
      else externalSignal.addEventListener('abort', abort, { once: true });
    }
    const midState = createAnthropicMidState(0);
    try {
      const parsed = await parseSSEStream(
        response,
        controller.signal,
        (reason) => controller.abort(reason),
        callbacks.onStreamDelta,
        callbacks.onToolCall,
        midState,
        callbacks.onTextDelta,
        knownToolNames,
        { onStreamItemStart: callbacks.onStreamItemStart }
      );
      assertAnthropicStreamNotEmpty(midState, parsed, 'Vertex AI Claude');
      return buildAnthropicTurnResult(parsed, { provider: this.name, useModel: model, opts });
    } finally {
      externalSignal?.removeEventListener?.('abort', abort);
    }
  }

  async listModels() {
    return [...GEMINI_MODELS, ...ANTHROPIC_MODELS].map((model) => ({ ...model, provider: this.name }));
  }

  async isAvailable() {
    try {
      return Boolean(this._projectId());
    } catch {
      return false;
    }
  }
}
