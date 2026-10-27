import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import test from 'node:test';

import { VertexAccessTokenManager, VERTEX_SCOPE, VERTEX_TOKEN_URI } from './vertex-ai-auth.mjs';
import { VertexAIProvider } from './vertex-ai.mjs';

function serviceAccount() {
  const { privateKey } = generateKeyPairSync('rsa', { modulusLength: 2048 });
  return {
    type: 'service_account',
    project_id: 'key-project',
    client_email: 'mixdog@test-project.iam.gserviceaccount.com',
    private_key: privateKey.export({ type: 'pkcs8', format: 'pem' }),
    token_uri: VERTEX_TOKEN_URI,
  };
}

function decodeJwtPart(part) {
  return JSON.parse(Buffer.from(part.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf8'));
}

function geminiSse(text = 'ok') {
  const chunk = {
    candidates: [{ content: { role: 'model', parts: [{ text }] }, finishReason: 'STOP' }],
    usageMetadata: { promptTokenCount: 7, candidatesTokenCount: 2, totalTokenCount: 9 },
  };
  return new Response(`data: ${JSON.stringify(chunk)}\n\n`, {
    headers: { 'content-type': 'text/event-stream' },
  });
}

function claudeSse(text = 'ok') {
  const frame = (event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
  return new Response(
    [
      frame({ type: 'message_start', message: { model: 'claude-sonnet-5-5', usage: { input_tokens: 5 } } }),
      frame({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
      frame({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }),
      frame({ type: 'content_block_stop', index: 0 }),
      frame({ type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 2 } }),
      frame({ type: 'message_stop' }),
    ].join(''),
    { headers: { 'content-type': 'text/event-stream' } }
  );
}

test('Vertex access tokens use an RS256 service-account JWT, cache, and refresh before expiry', async () => {
  const account = serviceAccount();
  let now = 1_800_000_000_000;
  const tokenRequests = [];
  const manager = new VertexAccessTokenManager({
    credentialResolver: () => JSON.stringify(account),
    now: () => now,
    fetchFn: async (url, init) => {
      tokenRequests.push({ url, init });
      return Response.json({ access_token: `token-${tokenRequests.length}`, expires_in: 3600 });
    },
  });

  assert.equal(await manager.getAccessToken(), 'token-1');
  assert.equal(await manager.getAccessToken(), 'token-1');
  assert.equal(tokenRequests.length, 1);
  const form = new URLSearchParams(tokenRequests[0].init.body);
  const [header, claims] = form.get('assertion').split('.');
  assert.equal(decodeJwtPart(header).alg, 'RS256');
  assert.equal(decodeJwtPart(claims).scope, VERTEX_SCOPE);
  assert.equal(decodeJwtPart(claims).aud, VERTEX_TOKEN_URI);

  now += 56 * 60_000;
  assert.equal(await manager.getAccessToken(), 'token-2');
  assert.equal(tokenRequests.length, 2);
});

test('Vertex Gemini routing reuses the Gemini body and bearer auth', async () => {
  const account = serviceAccount();
  const requests = [];
  const provider = new VertexAIProvider({
    apiKey: JSON.stringify(account),
    location: 'us-east5',
    projectId: 'override-project',
    fetchFn: async (url, init = {}) => {
      if (url === VERTEX_TOKEN_URI) return Response.json({ access_token: 'gcp-token', expires_in: 3600 });
      requests.push({ url, init });
      return geminiSse('gemini-ok');
    },
  });
  const result = await provider.send(
    [
      { role: 'system', content: 'system rule' },
      { role: 'user', content: 'hello' },
    ],
    'gemini-3.1-pro-preview',
    [],
    {}
  );
  assert.equal(result.content, 'gemini-ok');
  assert.match(
    requests[0].url,
    /^https:\/\/us-east5-aiplatform\.googleapis\.com\/v1\/projects\/override-project\/locations\/us-east5\/publishers\/google\/models\/gemini-3\.1-pro-preview:streamGenerateContent\?alt=sse$/
  );
  assert.equal(new Headers(requests[0].init.headers).get('authorization'), 'Bearer gcp-token');
  const body = JSON.parse(requests[0].init.body);
  assert.equal(body.systemInstruction, 'system rule');
  assert.equal(body.contents[0].role, 'user');
});

test('Vertex Claude routing sends Anthropic Messages body without model', async () => {
  const account = serviceAccount();
  const requests = [];
  const provider = new VertexAIProvider({
    apiKey: JSON.stringify(account),
    location: 'global',
    fetchFn: async (url, init = {}) => {
      if (url === VERTEX_TOKEN_URI) return Response.json({ access_token: 'gcp-token', expires_in: 3600 });
      requests.push({ url, init });
      return claudeSse('claude-ok');
    },
  });
  const result = await provider.send([{ role: 'user', content: 'hello' }], 'claude-sonnet-5-5', [], {});
  assert.equal(result.content, 'claude-ok');
  assert.equal(
    requests[0].url,
    'https://aiplatform.googleapis.com/v1/projects/key-project/locations/global/publishers/anthropic/models/claude-sonnet-5-5:streamRawPredict'
  );
  const body = JSON.parse(requests[0].init.body);
  assert.equal(body.anthropic_version, 'vertex-2023-10-16');
  assert.equal(Object.hasOwn(body, 'model'), false);
  assert.equal(body.stream, true);
});

test('Vertex authentication failures surface a clear provider error', async () => {
  const account = serviceAccount();
  const provider = new VertexAIProvider({
    apiKey: JSON.stringify(account),
    fetchFn: async (url) => {
      assert.equal(url, VERTEX_TOKEN_URI);
      return Response.json({ error: 'invalid_grant', error_description: 'service account rejected' }, { status: 400 });
    },
  });
  await assert.rejects(
    provider.send([{ role: 'user', content: 'hello' }], 'gemini-3.1-pro-preview', [], {}),
    (error) => error?.code === 'EVERTEXAUTH' && /Vertex AI authentication failed: service account rejected/.test(error.message)
  );
});
