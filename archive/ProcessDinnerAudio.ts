#!/usr/bin/env bun
/**
 * Process Tet Dinner Audio Clips
 *
 * Batch processes M4A audio files:
 *   1. Deepgram Nova-2 pre-recorded API (Vietnamese)
 *   2. Azure GPT-4o-mini translation (Vietnamese → English)
 *   3. Vocabulary extraction
 *   4. Outputs: markdown transcript + JSONL data + vocab update
 *
 * Usage: bun ProcessDinnerAudio.ts <folder-path>
 */

// Load .env from script directory
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

import { readdirSync } from "node:fs";
import { join, basename } from "node:path";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const AZURE_API_VERSION = "2025-01-01-preview";
const CHEAT_SHEET_PATH = `${process.env.HOME}/tet-cheat-sheet.md`;
const DG_API = "https://api.deepgram.com/v1/listen";

const TRANSLATE_SYSTEM = `You are a Vietnamese-to-English translator for a family Tet dinner conversation (Northern Vietnamese dialect).

Translate the Vietnamese text to natural English. Add brief cultural context only when the literal translation wouldn't convey the real meaning.

Format:
ENGLISH: [translation]
INTENT: [context, only if non-obvious — omit line entirely if translation is clear]

Be concise.`;

const VOCAB_SYSTEM = `Extract useful Vietnamese vocabulary from this dinner transcript for a learner (Northern dialect).

Output format (one per line):
| [Vietnamese] | "[phonetic]" | [English] |

Rules:
- Only useful beginner/intermediate words (food, family, Tet phrases, polite expressions)
- Skip common words (và, là, có, không, etc.)
- Northern pronunciation for phonetics
- Max 15 words per clip
- If nothing useful: NO_NEW_VOCAB`;

// ---------------------------------------------------------------------------
// Azure helper
// ---------------------------------------------------------------------------

function azureUrl(deployment: string): string {
  const endpoint = process.env.AZURE_OPENAI_ENDPOINT!.replace(/\/$/, "");
  return `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${AZURE_API_VERSION}`;
}

async function azureChat(system: string, user: string, maxTokens: number): Promise<string> {
  const res = await fetch(azureUrl("gpt-4o-mini"), {
    method: "POST",
    headers: { "Content-Type": "application/json", "api-key": process.env.AZURE_OPENAI_KEY! },
    body: JSON.stringify({
      messages: [{ role: "system", content: system }, { role: "user", content: user }],
      max_tokens: maxTokens,
      temperature: 0.3,
    }),
  });
  if (!res.ok) throw new Error(`Azure ${res.status}: ${await res.text()}`);
  const data = await res.json() as any;
  return data.choices?.[0]?.message?.content || "";
}

// ---------------------------------------------------------------------------
// Deepgram pre-recorded transcription
// ---------------------------------------------------------------------------

interface DgWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
  punctuated_word?: string;
}

interface DgUtterance {
  start: number;
  end: number;
  transcript: string;
  words: DgWord[];
  confidence: number;
}

async function transcribeFile(filePath: string): Promise<{ transcript: string; utterances: DgUtterance[] }> {
  const fileData = await Bun.file(filePath).arrayBuffer();

  const params = new URLSearchParams({
    model: "nova-2",
    language: "vi",
    smart_format: "true",
    punctuate: "true",
    utterances: "true",
    diarize: "true",
  });

  const res = await fetch(`${DG_API}?${params}`, {
    method: "POST",
    headers: {
      Authorization: `Token ${process.env.DEEPGRAM_API_KEY}`,
      "Content-Type": "audio/m4a",
    },
    body: fileData,
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Deepgram ${res.status}: ${err}`);
  }

  const data = await res.json() as any;
  const transcript = data.results?.channels?.[0]?.alternatives?.[0]?.transcript || "";
  const utterances: DgUtterance[] = data.results?.utterances || [];

  return { transcript, utterances };
}

// ---------------------------------------------------------------------------
// Translation
// ---------------------------------------------------------------------------

async function translateChunk(viText: string): Promise<{ english: string; intent?: string }> {
  if (!viText.trim()) return { english: "" };

  try {
    const text = await azureChat(TRANSLATE_SYSTEM, viText, 512);
    const englishMatch = text.match(/ENGLISH:\s*(.+?)(?:\n|$)/i);
    const intentMatch = text.match(/INTENT:\s*(.+?)(?:\n|$)/i);
    return {
      english: englishMatch?.[1]?.trim() || text.trim(),
      intent: intentMatch?.[1]?.trim() || undefined,
    };
  } catch (err: any) {
    return { english: `[Error: ${err.message}]` };
  }
}

// ---------------------------------------------------------------------------
// Time formatting
// ---------------------------------------------------------------------------

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fileTimeLabel(filename: string): string {
  // Extract time from filename like "02-18-2026 18.56(2).m4a" → "18:56"
  const match = filename.match(/(\d{2})\.(\d{2})(?:\(\d+\))?\.m4a$/);
  if (match) return `${match[1]}:${match[2]}`;
  return filename;
}

// ---------------------------------------------------------------------------
// Main processing
// ---------------------------------------------------------------------------

interface ProcessedUtterance {
  source_file: string;
  clip_time: string;
  offset_start: number;
  offset_end: number;
  speaker?: number;
  vietnamese: string;
  english: string;
  intent?: string;
}

async function processClip(filePath: string): Promise<{ utterances: ProcessedUtterance[]; rawVietnamese: string }> {
  const filename = basename(filePath);
  const clipTime = fileTimeLabel(filename);

  console.log(`  \x1b[36m🎙️  Transcribing: ${filename}\x1b[0m`);

  const { transcript, utterances } = await transcribeFile(filePath);

  if (!transcript.trim()) {
    console.log(`  \x1b[33m⚠️  No speech detected in ${filename}\x1b[0m`);
    return { utterances: [], rawVietnamese: "" };
  }

  console.log(`  \x1b[32m✓\x1b[0m  ${utterances.length} utterances, ${transcript.length} chars`);
  console.log(`  \x1b[2mTranslating...\x1b[0m`);

  // Translate utterances in batches of 5 for speed
  const processed: ProcessedUtterance[] = [];
  const batchSize = 5;

  for (let i = 0; i < utterances.length; i += batchSize) {
    const batch = utterances.slice(i, i + batchSize);
    const translations = await Promise.all(
      batch.map((u) => translateChunk(u.transcript))
    );

    for (let j = 0; j < batch.length; j++) {
      processed.push({
        source_file: filename,
        clip_time: clipTime,
        offset_start: batch[j].start,
        offset_end: batch[j].end,
        speaker: (batch[j] as any).speaker,
        vietnamese: batch[j].transcript,
        english: translations[j].english,
        intent: translations[j].intent,
      });
    }
  }

  console.log(`  \x1b[32m✓\x1b[0m  ${processed.length} utterances translated\n`);

  return { utterances: processed, rawVietnamese: transcript };
}

async function main(): Promise<void> {
  const folder = process.argv[2];
  if (!folder) {
    console.error("Usage: bun ProcessDinnerAudio.ts <folder-path>");
    process.exit(1);
  }

  // Preflight
  if (!process.env.DEEPGRAM_API_KEY) { console.error("  ✗ DEEPGRAM_API_KEY not set"); process.exit(1); }
  if (!process.env.AZURE_OPENAI_ENDPOINT || !process.env.AZURE_OPENAI_KEY) { console.error("  ✗ Azure keys not set"); process.exit(1); }

  // Find M4A files
  const files = readdirSync(folder)
    .filter((f) => f.endsWith(".m4a"))
    .sort()
    .map((f) => join(folder, f));

  if (files.length === 0) {
    console.error("  No M4A files found in", folder);
    process.exit(1);
  }

  console.log(`\n  \x1b[1m🇻🇳 Processing ${files.length} Tet Dinner Audio Clips\x1b[0m`);
  console.log(`  \x1b[2mPipeline: Deepgram Nova-2 (vi) → Azure GPT-4o-mini → Markdown + JSONL\x1b[0m\n`);

  const today = new Date().toISOString().slice(0, 10);
  const outputMd = join(folder, `tet-dinner-transcript-${today}.md`);
  const outputJsonl = join(folder, `tet-dinner-data-${today}.jsonl`);

  const allUtterances: ProcessedUtterance[] = [];
  const allRawVietnamese: string[] = [];

  // Process each clip sequentially (Deepgram rate limits)
  for (const file of files) {
    const result = await processClip(file);
    allUtterances.push(...result.utterances);
    if (result.rawVietnamese) allRawVietnamese.push(result.rawVietnamese);
  }

  // --- Write combined markdown ---
  let md = `---
date: ${today}
tool: ProcessDinnerAudio
engine: deepgram-nova-2 + azure-gpt-4o-mini
language: vi → en
clips: ${files.length}
total_utterances: ${allUtterances.length}
---

# Tet Dinner Transcript — ${today}

> ${files.length} audio clips processed | ${allUtterances.length} utterances
> Engine: Deepgram Nova-2 (Vietnamese) + Azure GPT-4o-mini (Translation)

`;

  let currentFile = "";
  for (const u of allUtterances) {
    if (u.source_file !== currentFile) {
      currentFile = u.source_file;
      md += `\n## Clip: ${u.source_file} (${u.clip_time})\n\n`;
    }
    md += `**[${formatTime(u.offset_start)}]**`;
    if (u.speaker !== undefined) md += ` Speaker ${u.speaker}`;
    md += `\n`;
    md += `- 🇻🇳 ${u.vietnamese}\n`;
    md += `- 🇬🇧 ${u.english}\n`;
    if (u.intent) md += `- 💡 _${u.intent}_\n`;
    md += `\n`;
  }

  await Bun.write(outputMd, md);
  console.log(`  \x1b[32m📄 Transcript:\x1b[0m ${outputMd}`);

  // --- Write JSONL ---
  const jsonlLines = allUtterances.map((u) => JSON.stringify(u)).join("\n") + "\n";
  await Bun.write(outputJsonl, jsonlLines);
  console.log(`  \x1b[32m📊 Data (JSONL):\x1b[0m ${outputJsonl}`);

  // --- Extract vocabulary ---
  if (allRawVietnamese.length > 0) {
    console.log(`  \x1b[2mExtracting vocabulary...\x1b[0m`);
    const combinedVi = allRawVietnamese.join("\n\n");
    // Chunk if too long (keep under 4000 chars for the prompt)
    const chunks = combinedVi.length > 4000
      ? [combinedVi.slice(0, 4000), combinedVi.slice(4000, 8000)].filter(Boolean)
      : [combinedVi];

    const vocabResults = await Promise.all(
      chunks.map((chunk) => azureChat(VOCAB_SYSTEM, chunk, 1024))
    );

    const vocabLines = vocabResults
      .filter((v) => !v.includes("NO_NEW_VOCAB"))
      .join("\n")
      .trim();

    if (vocabLines) {
      // Append to cheat sheet
      const cheatSheet = Bun.file(CHEAT_SHEET_PATH);
      if (await cheatSheet.exists()) {
        const existing = await cheatSheet.text();
        const sectionHeader = "## Learned at Dinner";

        if (existing.includes(sectionHeader)) {
          await Bun.write(CHEAT_SHEET_PATH, existing.trimEnd() + "\n" + vocabLines + "\n");
        } else {
          await Bun.write(CHEAT_SHEET_PATH, existing.trimEnd() + `\n\n---\n\n${sectionHeader}\n\n> Words and phrases picked up during Tet dinner — auto-captured\n\n| Vietnamese | Sounds like | English |\n|---|---|---|\n${vocabLines}\n`);
        }
        console.log(`  \x1b[35m📚 Vocab added to tet-cheat-sheet.md\x1b[0m`);
      }

      // Also save vocab to the output folder
      await Bun.write(join(folder, `vocab-${today}.md`), `# Vocabulary from Tet Dinner — ${today}\n\n| Vietnamese | Sounds like | English |\n|---|---|---|\n${vocabLines}\n`);
      console.log(`  \x1b[35m📚 Vocab file:\x1b[0m ${join(folder, `vocab-${today}.md`)}`);
    } else {
      console.log(`  \x1b[33m⚠️  No new vocabulary extracted\x1b[0m`);
    }
  }

  // --- Summary ---
  console.log(`\n  \x1b[1m✅ Done!\x1b[0m`);
  console.log(`  \x1b[2m${files.length} clips → ${allUtterances.length} utterances → markdown + JSONL + vocab\x1b[0m\n`);
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(`\n  \x1b[31mFatal: ${err.message}\x1b[0m`);
    process.exit(1);
  });
}
