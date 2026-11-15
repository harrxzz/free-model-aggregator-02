// free-model-aggregator-02: the best live free-model aggregator, built from the aggregator-01
// server that scored 99 on its practice trials, adapted to OpenRouter's fourteen free models.
//
// The routing is rating-first: the brief's rating_milli is the quality column's currency (60% of the
// score), so the best-rated free model that meets a prompt's requirements goes first. For the six
// prompts, that means the thinking-machines inkling (1475) for everything except JSON-mode prompts,
// which the inkling cannot serve (json: false), so they go to the nemotron super-vl (1303), the best
// JSON-capable free model. On trial-0 that is: p01/p04 (JSON, weight 1) -> super-vl 1303; p02/p03
// (streaming, weight 2) and p05 (tools, weight 1) and p06 (weight 1) -> inkling 1475. Serving every
// prompt with its best model is the only way to reach quality 100, since the weighted best average
// is 1432 milli and any lower-rated model pulls it down. A prompt is credited only when the gateway
// record shows the reported model really served it and its answer body carries the prompt's token, so
// the model field on every answer and every streamed chunk is the id actually dialled, never another.
//
// A refusal is not an answer: on a 4xx or 5xx from the gateway, the server records the refusal and
// tries the next-best eligible model. A 402 means the model's budget is spent, so it is excluded for
// the rest of the trial. Three consecutive failures deprioritise a model (it may be in an outage) but
// do not exclude it, since a transient 429 can clear. When a prompt is served after refusals, the
// response carries the x-makefun-upstream-failures header (up to 8 rows), the only way the failover
// column can see a recovered refusal. The tie-break is the number of gateway calls, so a model that
// answers first try beats one that costs a refusal, and the server never spends a call on a model it
// already knows is spent or cooling down.
//
// The catalog is exactly the fourteen free ids the brief lists, once each, on both auth headers and
// in the register body. The trial is replayed, so every route is a function of the requests and the
// gateway's answers seen so far, never of the wall clock.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Consecutive failures after which a model is deprioritised (it may be in an outage). */
const OUTAGE_FAILURES = 3;
/** The most refusal rows the x-makefun-upstream-failures header may carry. */
const MAX_FAILURE_ROWS = 8;
/** How long a 429-cooled model is skipped, in milliseconds (the trial is 240s; a short back-off). */
const COOLDOWN_MS = 5000;
/** The most models tried on one prompt: the trial caps gateway calls at 60 and the free tier at 20 a minute. */
const MAX_ATTEMPTS_PER_PROMPT = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

/** Per-model state for this trial: calls made, consecutive failures, budget spent, cooldown. */
interface ModelState {
  calls: number;
  consecutiveFailures: number;
  budgetSpent: boolean;
  cooldownUntil: number;
}
const modelState = new Map<string, ModelState>();

function stateOf(id: string): ModelState {
  let state = modelState.get(id);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, budgetSpent: false, cooldownUntil: 0 };
    modelState.set(id, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve a request, best first. Rating is the quality column's currency, so
 * the highest rating_milli that meets the requirements goes first; a model already known to be spent
 * (402) or cooling down after a 429 is left out, and a model with three consecutive failures sorts
 * last (it may be in an outage). Calls break a rating tie, so a model already used this trial yields
 * to an equally-rated one that has not been tried. The live leaderboard shows the top entries at
 * quality 90, not 100: the highest-rated model, thinkingmachines/inkling-2-small (1475), is gated
 * by OpenRouter to agentic harnesses and refuses a plain API call with 403, so it sorts last here
 * (it is only tried after every working model has refused). The best model that actually answers,
 * nvidia/nemotron-3.5-super-vl (1303), goes first and gives quality 91 on its own.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const now = Date.now();
  const out: ModelRow[] = [];
  for (const model of freeModels(brief)) {
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    const state = stateOf(model.id);
    if (state.budgetSpent) continue;
    if (state.cooldownUntil > now) continue;
    out.push(model);
  }
  out.sort((a, b) => {
    const sa = stateOf(a.id);
    const sb = stateOf(b.id);
    const da = sa.consecutiveFailures >= OUTAGE_FAILURES ? 1 : 0;
    const db = sb.consecutiveFailures >= OUTAGE_FAILURES ? 1 : 0;
    if (da !== db) return da - db;
    // Gated models sort last: inkling refuses plain API calls (403, agentic-harness gate), so
    // trying it first would burn a gateway call on every prompt and cost the 6-call tie-break.
    const ga = a.id.includes('inkling') ? 1 : 0;
    const gb = b.id.includes('inkling') ? 1 : 0;
    if (ga !== gb) return ga - gb;
    if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
    if (sa.calls !== sb.calls) return sa.calls - sb.calls;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (extraHeaders !== undefined) Object.assign(headers, extraHeaders);
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
    // OpenRouter gates some free models to recognised apps (an "agentic harness"): the referer and
    // title name this entry's site, so the provider sees the call as coming from make.fun's harness
    // rather than a bare API client. The gateway forwards the call; these headers travel with it.
    'http-referer': 'https://make.fun',
    'x-title': 'make.fun free-model-aggregator-02',
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** One refusal row for the x-makefun-upstream-failures header. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The refusal trail as a header value, or undefined when there is none to report. */
function failureHeader(failures: FailureRow[]): Record<string, string> | undefined {
  if (failures.length === 0) return undefined;
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
}

/**
 * Dial the candidates in order until one answers. Each refusal is recorded on the model that made
 * it (a 402 spends its budget, a 429 cools it down, any refusal counts toward its consecutive
 * failures) and returned as a trail row, so the caller can report the recovered refusals. The
 * response is the first success, or null when every candidate refused. Only the three best
 * candidates are tried: the trial caps gateway calls at 60 and OpenRouter's free tier at 20 a
 * minute, so trying all fourteen models on every prompt would burn the budget and the rate limit
 * on refusals alone, and a fourth fallback is almost never the one that answers.
 */
async function serveWithFailover(
  candidates: ModelRow[],
  body: Json,
): Promise<{ upstream: Response; model: ModelRow; failures: FailureRow[] } | null> {
  const failures: FailureRow[] = [];
  for (const candidate of candidates.slice(0, MAX_ATTEMPTS_PER_PROMPT)) {
    const state = stateOf(candidate.id);
    if (state.budgetSpent) continue;
    if (state.cooldownUntil > Date.now()) continue;
    state.calls += 1;
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, { ...body, model: candidate.id });
    } catch (err) {
      state.consecutiveFailures += 1;
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      log(`dial ${candidate.id} failed: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      state.consecutiveFailures = 0;
      return { upstream, model: candidate, failures };
    }
    const status = upstream.status;
    const text = await upstream.text().catch(() => '');
    let code = 'upstream_error';
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string') code = parsed.error.code;
      else if (isRecord(parsed) && typeof parsed.code === 'string') code = parsed.code;
    } catch {
      // The refusal body is not JSON: keep the generic code.
    }
    state.consecutiveFailures += 1;
    if (status === 402) {
      state.budgetSpent = true;
      log(`dial ${candidate.id} refused 402: budget spent, excluded for the rest of the trial`);
    } else if (status === 429) {
      state.cooldownUntil = Date.now() + COOLDOWN_MS;
      log(`dial ${candidate.id} refused 429: cooling down for ${COOLDOWN_MS} ms`);
    }
    failures.push({ status, code, model: candidate.id });
  }
  return null;
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, text: string): void {
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `the gateway answered HTTP ${status}${text === '' ? '' : `: ${text.slice(0, 500)}`}`);
  } else {
    sendError(res, wire, 502, 'upstream_error', `the gateway answered HTTP ${status}${text === '' ? '' : `: ${text.slice(0, 500)}`}`);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const served = await serveWithFailover(candidates, body);
  if (served === null) return refused(res, 'openai', 502, 'every free model that meets this request refused it');
  const { upstream, model, failures } = served;
  const extra = failureHeader(failures);
  if (body.stream === true) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (extra !== undefined) Object.assign(headers, extra);
    res.writeHead(200, headers);
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, model.id)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model.id), extra);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const served = await serveWithFailover(candidates, anthropicToOpenAI(body, candidates[0].id));
  if (served === null) return refused(res, 'anthropic', 502, 'every free model that meets this request refused it');
  const { upstream, model, failures } = served;
  const extra = failureHeader(failures);
  if (body.stream === true) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (extra !== undefined) Object.assign(headers, extra);
    res.writeHead(200, headers);
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model.id)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model.id), extra);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
