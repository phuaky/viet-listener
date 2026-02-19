#!/usr/bin/env bun
/**
 * ============================================================================
 * TRANSCRIBE LIVE (Vietnamese → English)
 * ============================================================================
 *
 * Real-time Vietnamese speech transcription + AI translation to English.
 * Built for Tet dinner with northern Vietnamese in-laws.
 *
 * Pipeline: Mic → sox → Deepgram Nova-2 (vi) → Azure GPT-4o-mini (translate) → Display + Log
 *
 * USAGE:
 *   bun TranscribeLiveVi.ts
 *
 * SIGNALS:
 *   SIGUSR1  - Cut current segment, extract vocab, save to markdown
 *   SIGINT   - Process remaining buffer and exit gracefully
 *   SIGTERM  - Process remaining buffer and exit gracefully
 *
 * REQUIREMENTS:
 *   - sox installed (brew install sox)
 *   - DEEPGRAM_API_KEY environment variable
 *   - AZURE_OPENAI_ENDPOINT environment variable
 *   - AZURE_OPENAI_KEY environment variable
 *
 * OUTPUT:
 *   transcript-vi-YYYY-MM-DD.md in current directory
 *   New vocab appended to ~/tet-cheat-sheet.md on each cut
 *
 * ============================================================================
 */

// Load .env from script directory (Bun auto-loads from CWD, but script may run from elsewhere)
const scriptDir = import.meta.dir;
const envFile = Bun.file(`${scriptDir}/.env`);
if (await envFile.exists()) {
  const lines = (await envFile.text()).split("\n");
  for (const line of lines) {
    const match = line.match(/^([A-Z_]+)=(.+)$/);
    if (match && !process.env[match[1]]) {
      process.env[match[1]] = match[2];
    }
  }
}

import { createClient, LiveTranscriptionEvents } from "@deepgram/sdk";
import { spawn, type ChildProcess } from "child_process";
import { unlinkSync } from "node:fs";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PID_FILE = "/tmp/transcribe-live.pid"; // same as English version so Hammerspoon Cmd+Shift+T works
const CHEAT_SHEET_PATH = `${process.env.HOME}/tet-cheat-sheet.md`;
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_BASE_DELAY_MS = 1000;

const TRANSLATE_SYSTEM_PROMPT = `You are a Vietnamese-to-English translator for a family Tet dinner conversation (Northern Vietnamese dialect).

Your job:
1. Translate the Vietnamese text to natural English
2. Add brief context/intent when the literal translation wouldn't convey the real meaning
3. Note any culturally significant phrases (Tet wishes, family honorifics, food references)

Format your response as:
ENGLISH: [natural English translation]
INTENT: [what they really mean / cultural context, only if non-obvious — skip if translation is clear enough]

Be concise. This runs in real-time so speed matters.`;

const VOCAB_EXTRACT_PROMPT = `You are a vocabulary extractor for a Vietnamese language learner (Northern dialect).

Given a Vietnamese transcript segment, extract NEW vocabulary words/phrases that would be useful to memorize.

For each word/phrase, output EXACTLY this format (one per line):
| [Vietnamese] | "[phonetic]" | [English meaning] |

Rules:
- Only include words that a beginner/intermediate learner would benefit from
- Skip common words they likely already know (và, là, có, không, etc.)
- Include food items, family terms, Tet-specific phrases, and polite expressions
- Use Northern pronunciation for phonetics
- Maximum 10 words per segment
- If no new useful vocabulary, output: NO_NEW_VOCAB`;

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const RED = "\x1b[31m";
const CYAN = "\x1b[36m";
const BOLD = "\x1b[1m";
const MAGENTA = "\x1b[35m";

function dimText(text: string): string { return `${DIM}${text}${RESET}`; }
function greenText(text: string): string { return `${GREEN}${text}${RESET}`; }
function yellowText(text: string): string { return `${YELLOW}${text}${RESET}`; }
function redText(text: string): string { return `${RED}${text}${RESET}`; }
function cyanText(text: string): string { return `${CYAN}${text}${RESET}`; }
function boldText(text: string): string { return `${BOLD}${text}${RESET}`; }
function magentaText(text: string): string { return `${MAGENTA}${text}${RESET}`; }

// ---------------------------------------------------------------------------
// Date/time helpers
// ---------------------------------------------------------------------------

function todayDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function nowTimestamp(): string {
  return new Date().toTimeString().slice(0, 8);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface TranscriptEntry {
  vietnamese: string;
  english: string;
  intent?: string;
  timestamp: string;
}

interface AppState {
  transcriptBuffer: string[];
  translatedEntries: TranscriptEntry[];
  segmentNumber: number;
  outputFile: string;
  soxProcess: ChildProcess | null;
  isShuttingDown: boolean;
  reconnectAttempts: number;
  deepgramConnection: ReturnType<ReturnType<typeof createClient>["listen"]["live"]> | null;
  lastInterimLine: string;
  processingPromises: Promise<void>[];
  translationQueue: Promise<void>[];
}

const state: AppState = {
  transcriptBuffer: [],
  translatedEntries: [],
  segmentNumber: 0,
  outputFile: `transcript-vi-${todayDate()}.md`,
  soxProcess: null,
  isShuttingDown: false,
  reconnectAttempts: 0,
  deepgramConnection: null,
  lastInterimLine: "",
  processingPromises: [],
  translationQueue: [],
};

// ---------------------------------------------------------------------------
// Azure GPT-4o-mini translation (fastest + cheapest on Azure)
// ---------------------------------------------------------------------------

const AZURE_API_VERSION = "2025-01-01-preview";

function azureUrl(deployment: string): string {
  const endpoint = process.env.AZURE_OPENAI_ENDPOINT!.replace(/\/$/, "");
  return `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${AZURE_API_VERSION}`;
}

async function azureChat(systemPrompt: string, userPrompt: string, maxTokens: number): Promise<string> {
  const response = await fetch(azureUrl("gpt-4o-mini"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "api-key": process.env.AZURE_OPENAI_KEY!,
    },
    body: JSON.stringify({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      max_tokens: maxTokens,
      temperature: 0.3,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Azure API ${response.status}: ${err}`);
  }

  const data = await response.json() as any;
  return data.choices?.[0]?.message?.content || "";
}

async function translateViToEn(vietnameseText: string): Promise<{ english: string; intent?: string }> {
  try {
    const text = await azureChat(TRANSLATE_SYSTEM_PROMPT, vietnameseText, 256);

    const englishMatch = text.match(/ENGLISH:\s*(.+?)(?:\n|$)/i);
    const intentMatch = text.match(/INTENT:\s*(.+?)(?:\n|$)/i);

    return {
      english: englishMatch?.[1]?.trim() || text.trim(),
      intent: intentMatch?.[1]?.trim() || undefined,
    };
  } catch (err: any) {
    return { english: `[Translation error: ${err.message}]` };
  }
}

async function extractVocab(vietnameseText: string): Promise<string | null> {
  try {
    const text = await azureChat(VOCAB_EXTRACT_PROMPT, vietnameseText, 512);
    if (text.includes("NO_NEW_VOCAB")) return null;
    return text.trim();
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// PID file management
// ---------------------------------------------------------------------------

function writePidFile(): void {
  Bun.write(PID_FILE, String(process.pid));
}

function removePidFile(): void {
  try { unlinkSync(PID_FILE); } catch { /* may already be gone */ }
}

// ---------------------------------------------------------------------------
// Markdown file management
// ---------------------------------------------------------------------------

async function ensureMarkdownHeader(): Promise<void> {
  const file = Bun.file(state.outputFile);
  if (!(await file.exists())) {
    await Bun.write(state.outputFile, `---
date: ${todayDate()}
tool: TranscribeLiveVi
engine: deepgram-nova-2
language: vi → en
mode: northern-vietnamese
---

# Vietnamese Live Transcript — ${todayDate()}

> Real-time Vietnamese → English transcription for Tet dinner
> Engine: Deepgram Nova-2 (Vietnamese) + Azure GPT-4o-mini (Translation)

`);
  }
}

async function appendSegment(
  segNum: number,
  timestamp: string,
  entries: TranscriptEntry[],
  rawVietnamese: string,
): Promise<void> {
  let block = `## Segment ${segNum} — ${timestamp}\n\n`;

  // Conversation log
  block += "### Conversation\n\n";
  for (const entry of entries) {
    block += `**[${entry.timestamp}]**\n`;
    block += `- 🇻🇳 ${entry.vietnamese}\n`;
    block += `- 🇬🇧 ${entry.english}\n`;
    if (entry.intent) {
      block += `- 💡 _${entry.intent}_\n`;
    }
    block += "\n";
  }

  // Raw Vietnamese for reference
  block += "### Raw Vietnamese\n";
  block += `${rawVietnamese}\n\n`;
  block += "---\n\n";

  const existing = await Bun.file(state.outputFile).text();
  await Bun.write(state.outputFile, existing + block);
}

async function appendVocabToCheatSheet(vocabLines: string): Promise<void> {
  const file = Bun.file(CHEAT_SHEET_PATH);
  if (!(await file.exists())) return;

  const existing = await file.text();

  // Check if we already have a "Learned at Dinner" section
  const sectionHeader = "## Learned at Dinner";
  let content: string;

  if (existing.includes(sectionHeader)) {
    // Append to existing section (before the last line)
    content = existing.trimEnd() + "\n" + vocabLines + "\n";
  } else {
    // Create new section at the end
    content = existing.trimEnd() + `\n\n---\n\n${sectionHeader}\n\n> Words and phrases picked up during Tet dinner — auto-captured by TranscribeLiveVi\n\n| Vietnamese | Sounds like | English |\n|---|---|---|\n${vocabLines}\n`;
  }

  await Bun.write(CHEAT_SHEET_PATH, content);
}

// ---------------------------------------------------------------------------
// Cut handler (SIGUSR1)
// ---------------------------------------------------------------------------

async function handleCut(): Promise<void> {
  const bufferSnapshot = state.transcriptBuffer.splice(0, state.transcriptBuffer.length);
  const entriesSnapshot = state.translatedEntries.splice(0, state.translatedEntries.length);

  if (bufferSnapshot.length === 0 && entriesSnapshot.length === 0) {
    console.log(yellowText("\n  No transcript to cut — buffer is empty.\n"));
    return;
  }

  state.segmentNumber++;
  const segNum = state.segmentNumber;
  const timestamp = nowTimestamp();
  const rawVietnamese = bufferSnapshot.join(" ");

  // Clear interim line
  if (state.lastInterimLine) {
    process.stdout.write("\r" + " ".repeat(process.stdout.columns || 80) + "\r");
    state.lastInterimLine = "";
  }

  console.log(cyanText(`\n  ✂️  CUT — Processing segment ${segNum}...\n`));

  const processingPromise = (async () => {
    try {
      await ensureMarkdownHeader();
      await appendSegment(segNum, timestamp, entriesSnapshot, rawVietnamese);
      console.log(greenText(`  ✅ Segment ${segNum} saved to ${state.outputFile}\n`));

      // Extract and append vocabulary
      if (rawVietnamese.trim()) {
        const vocab = await extractVocab(rawVietnamese);
        if (vocab) {
          await appendVocabToCheatSheet(vocab);
          console.log(magentaText(`  📚 New vocab added to tet-cheat-sheet.md\n`));
        }
      }
    } catch (err: any) {
      console.error(redText(`  Error processing segment ${segNum}: ${err.message}`));
    }
  })();

  state.processingPromises.push(processingPromise);
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

async function shutdown(): Promise<void> {
  if (state.isShuttingDown) return;
  state.isShuttingDown = true;

  console.log(yellowText("\n\n  Shutting down...\n"));

  // Wait for pending translations
  if (state.translationQueue.length > 0) {
    console.log(dimText("  Waiting for pending translations..."));
    await Promise.allSettled(state.translationQueue);
  }

  // Process remaining buffer
  if (state.transcriptBuffer.length > 0 || state.translatedEntries.length > 0) {
    console.log(dimText("  Processing remaining buffer..."));
    await handleCut();
  }

  // Wait for all pending processing
  if (state.processingPromises.length > 0) {
    console.log(dimText("  Waiting for pending segments..."));
    await Promise.allSettled(state.processingPromises);
  }

  // Close Deepgram
  if (state.deepgramConnection) {
    try { state.deepgramConnection.disconnect(); } catch { /* */ }
    state.deepgramConnection = null;
  }

  // Kill sox
  if (state.soxProcess && !state.soxProcess.killed) {
    state.soxProcess.kill("SIGTERM");
    state.soxProcess = null;
  }

  removePidFile();
  console.log(greenText("  Clean shutdown complete.\n"));
  process.exit(0);
}

// ---------------------------------------------------------------------------
// Preflight checks
// ---------------------------------------------------------------------------

function preflightChecks(): boolean {
  let ok = true;

  if (!process.env.DEEPGRAM_API_KEY) {
    console.error(redText("  ✗ DEEPGRAM_API_KEY not set"));
    console.error(dimText("    Get key from https://console.deepgram.com"));
    ok = false;
  }

  if (!process.env.AZURE_OPENAI_ENDPOINT || !process.env.AZURE_OPENAI_KEY) {
    console.error(redText("  ✗ AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_KEY not set"));
    console.error(dimText("    Needed for Vietnamese → English translation via Azure GPT-4o-mini"));
    ok = false;
  }

  try {
    const which = Bun.spawnSync(["which", "rec"]);
    if (which.exitCode !== 0) throw new Error("rec not found");
  } catch {
    console.error(redText("  ✗ sox (rec command) not installed"));
    console.error(dimText("    Install with: brew install sox"));
    ok = false;
  }

  return ok;
}

// ---------------------------------------------------------------------------
// Audio capture via sox
// ---------------------------------------------------------------------------

function startAudioCapture(): ChildProcess {
  const soxProc = spawn("rec", [
    "-q",
    "-t", "raw",
    "-b", "16",
    "-e", "signed-integer",
    "-r", "16000",
    "-c", "1",
    "-",
  ], {
    stdio: ["ignore", "pipe", "pipe"],
  });

  soxProc.stderr?.on("data", (data: Buffer) => {
    const msg = data.toString().trim();
    if (msg && !state.isShuttingDown && !msg.includes("WARN")) {
      console.error(dimText(`  [sox] ${msg}`));
    }
  });

  soxProc.on("error", (err: Error) => {
    if (!state.isShuttingDown) {
      console.error(redText(`  Sox error: ${err.message}`));
      shutdown();
    }
  });

  soxProc.on("exit", (code, signal) => {
    if (!state.isShuttingDown) {
      console.error(yellowText(`  Sox exited (code=${code}, signal=${signal})`));
      shutdown();
    }
  });

  return soxProc;
}

// ---------------------------------------------------------------------------
// Deepgram connection (Vietnamese)
// ---------------------------------------------------------------------------

function createDeepgramConnection() {
  const client = createClient(process.env.DEEPGRAM_API_KEY!);

  const connection = client.listen.live({
    model: "nova-2",
    language: "vi",           // Vietnamese!
    smart_format: true,
    interim_results: true,
    utterance_end_ms: 1200,   // slightly longer for Vietnamese phrase boundaries
    punctuate: true,
    encoding: "linear16",
    sample_rate: 16000,
    channels: 1,
  });

  connection.on(LiveTranscriptionEvents.Open, () => {
    state.reconnectAttempts = 0;
    console.log(greenText("  ✓ Deepgram connected (Vietnamese, Nova-2)\n"));
    console.log(
      boldText("  🎙️  Listening... ") +
      dimText("(Cmd+Shift+T to cut, Ctrl+C to stop)\n"),
    );
  });

  connection.on(LiveTranscriptionEvents.Transcript, (data: any) => {
    const transcript = data.channel?.alternatives?.[0]?.transcript;
    if (!transcript || transcript.trim() === "") return;

    if (data.is_final) {
      const viText = transcript.trim();
      state.transcriptBuffer.push(viText);

      // Clear interim
      if (state.lastInterimLine) {
        process.stdout.write("\r" + " ".repeat(process.stdout.columns || 80) + "\r");
        state.lastInterimLine = "";
      }

      // Show Vietnamese immediately
      process.stdout.write(cyanText("  🇻🇳 ") + viText + "\n");

      // Translate async — don't block audio
      const translatePromise = (async () => {
        const result = await translateViToEn(viText);
        const entry: TranscriptEntry = {
          vietnamese: viText,
          english: result.english,
          intent: result.intent,
          timestamp: nowTimestamp(),
        };
        state.translatedEntries.push(entry);

        // Show English translation
        process.stdout.write(greenText("  🇬🇧 ") + result.english + "\n");
        if (result.intent) {
          process.stdout.write(dimText(`  💡 ${result.intent}`) + "\n");
        }
        process.stdout.write("\n");
      })();

      state.translationQueue.push(translatePromise);
    } else {
      // Interim — show dimmed Vietnamese
      const interimText = transcript.trim();
      if (interimText) {
        if (state.lastInterimLine) {
          process.stdout.write("\r" + " ".repeat(process.stdout.columns || 80) + "\r");
        }
        const displayText = `  ${dimText("🇻🇳 " + interimText)}`;
        process.stdout.write(`\r${displayText}`);
        state.lastInterimLine = displayText;
      }
    }
  });

  connection.on(LiveTranscriptionEvents.Error, (err: any) => {
    console.error(redText(`\n  Deepgram error: ${err.message || JSON.stringify(err)}`));
    if (!state.isShuttingDown) attemptReconnect();
  });

  connection.on(LiveTranscriptionEvents.Close, () => {
    if (!state.isShuttingDown) {
      console.log(yellowText("\n  Deepgram connection closed."));
      attemptReconnect();
    }
  });

  return connection;
}

// ---------------------------------------------------------------------------
// Reconnection
// ---------------------------------------------------------------------------

function attemptReconnect(): void {
  if (state.isShuttingDown) return;
  state.reconnectAttempts++;

  if (state.reconnectAttempts > MAX_RECONNECT_ATTEMPTS) {
    console.error(redText(`  Max reconnects (${MAX_RECONNECT_ATTEMPTS}) exceeded.`));
    shutdown();
    return;
  }

  const delay = RECONNECT_BASE_DELAY_MS * Math.pow(2, state.reconnectAttempts - 1);
  console.log(yellowText(`  Reconnecting in ${delay}ms (${state.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`));

  setTimeout(() => {
    if (state.isShuttingDown) return;
    try {
      state.deepgramConnection = createDeepgramConnection();
      pipeAudioToDeepgram();
    } catch (err: any) {
      console.error(redText(`  Reconnect failed: ${err.message}`));
      attemptReconnect();
    }
  }, delay);
}

// ---------------------------------------------------------------------------
// Pipe audio
// ---------------------------------------------------------------------------

function pipeAudioToDeepgram(): void {
  if (!state.soxProcess?.stdout || !state.deepgramConnection) return;

  state.soxProcess.stdout.on("data", (audioChunk: Buffer) => {
    if (state.isShuttingDown) return;
    try {
      if (state.deepgramConnection && state.deepgramConnection.isConnected()) {
        state.deepgramConnection.send(audioChunk);
      }
    } catch (err: any) {
      if (err.code !== "EPIPE" && !state.isShuttingDown) {
        console.error(dimText(`  Send error: ${err.message}`));
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log(boldText("\n  🇻🇳 TranscribeLive Vietnamese → English"));
  console.log(dimText("  Real-time Vietnamese transcription + AI translation\n"));
  console.log(dimText("  Pipeline: Mic → Deepgram Nova-2 (vi) → Azure GPT-4o-mini (translate) → Log\n"));

  if (!preflightChecks()) {
    process.exit(1);
  }

  writePidFile();
  console.log(dimText(`  PID: ${process.pid}`));
  console.log(dimText(`  Output: ${state.outputFile}`));
  console.log(dimText(`  Vocab: ${CHEAT_SHEET_PATH}\n`));

  // Signal handlers
  process.on("SIGUSR1", () => { handleCut(); });
  process.on("SIGINT", () => { shutdown(); });
  process.on("SIGTERM", () => { shutdown(); });

  // Start
  console.log(dimText("  Starting microphone..."));
  state.soxProcess = startAudioCapture();

  console.log(dimText("  Connecting to Deepgram (Nova-2, Vietnamese)..."));
  state.deepgramConnection = createDeepgramConnection();

  state.deepgramConnection.on(LiveTranscriptionEvents.Open, () => {
    pipeAudioToDeepgram();
  });
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(redText(`\n  Fatal: ${err.message}`));
    removePidFile();
    process.exit(1);
  });
}
