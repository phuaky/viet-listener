#!/usr/bin/env bun
/**
 * Process Audio — Gemini-based Vietnamese dinner audio processor
 *
 * Pipeline per file:
 *   1. Read M4A → base64 encode
 *   2. POST to Gemini 2.5 Flash (structured JSON output)
 *   3. Parse response → JSONL + markdown transcript
 *   4. Extract vocabulary via dictionary longest-match
 *   5. Update word-frequency.json
 *   6. Generate study sheet (top 30 words)
 *
 * Usage: bun run ProcessAudio.ts <folder-or-file>
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

import { readdirSync, existsSync, mkdirSync, unlinkSync } from "node:fs";
import { join, basename } from "node:path";
import { tmpdir } from "node:os";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

const OUTPUT_DIR = "tet-dinner/processing/gemini-v2";
const WORD_FREQ_PATH = "tet-dinner/data/word-frequency.json";
const MAX_FILE_SIZE_MB = 20; // Split files larger than this
const CHUNK_DURATION_SEC = 600; // 10-minute chunks

const GEMINI_SYSTEM_PROMPT = `You are transcribing a Vietnamese family dinner conversation recorded on a phone. The audio is from a Tet (Lunar New Year) dinner in Hanoi, Vietnam.

Family members present:
- **Bo** (Bố/Dad): Older Vietnamese man, speaks mostly Vietnamese (Northern dialect)
- **Daughter** (Con gái): Younger Vietnamese woman, switches between Vietnamese and English
- **Guy**: English-speaking man (the daughter's partner), speaks some Vietnamese

Instructions:
- Transcribe ALL speech — both Vietnamese and English portions
- Identify speakers by matching voice characteristics to the family members above
- For Vietnamese utterances: provide the Vietnamese text and a natural English translation
- For English utterances: provide the English text as both vietnamese and english fields
- Include brief intent/context when the literal translation doesn't convey meaning
- Preserve the natural conversational flow including interruptions and overlapping speech
- Use Northern Vietnamese dialect spelling conventions`;

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

interface GeminiUtterance {
  speaker_id: string;
  speaker_name: string;
  vietnamese: string;
  english: string;
  intent?: string;
}

interface GeminiResponse {
  session_summary?: string;
  speakers?: Array<{ id: string; name: string; role?: string }>;
  utterances: GeminiUtterance[];
}

interface ProcessedUtterance {
  source_file: string;
  clip_time: string;
  speaker_name: string;
  vietnamese: string;
  english: string;
  intent?: string;
}

// ---------------------------------------------------------------------------
// Dictionary loading + Vietnamese segmentation (from server.ts)
// ---------------------------------------------------------------------------

const dictFile = Bun.file(join(scriptDir, "dictionary.json"));
let dictionary: Record<string, string> = {};
try {
  dictionary = await dictFile.json();
} catch {
  console.warn("  dictionary.json not found, vocab extraction will be limited");
}

function segmentVietnamese(text: string): string[] {
  const cleaned = text.trim()
    .replace(/[.,!?;:"""''()[\]{}]/g, ' ')
    .replace(/\s+/g, ' ');

  const syllables = cleaned.split(' ').filter(s => s.length > 0);
  const words: string[] = [];

  let i = 0;
  while (i < syllables.length) {
    let matched = false;

    // Try longest match first (up to 4 syllables)
    for (let len = Math.min(4, syllables.length - i); len > 1; len--) {
      const compound = syllables.slice(i, i + len).join(' ');
      const lower = compound.toLowerCase();
      if (dictionary[lower] || dictionary[compound]) {
        words.push(compound);
        i += len;
        matched = true;
        break;
      }
    }

    if (!matched) {
      words.push(syllables[i]);
      i++;
    }
  }

  return words;
}

// ---------------------------------------------------------------------------
// Time extraction from filename
// ---------------------------------------------------------------------------

function fileTimeLabel(filename: string): string {
  // Extract time from filename like "02-18-2026 18.56(2).m4a" → "18:56"
  const match = filename.match(/(\d{2})\.(\d{2})(?:\(\d+\))?\.m4a$/);
  if (match) return `${match[1]}:${match[2]}`;
  return filename;
}

// ---------------------------------------------------------------------------
// Gemini API call
// ---------------------------------------------------------------------------

const responseSchema = {
  type: "OBJECT",
  properties: {
    session_summary: { type: "STRING" },
    speakers: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        properties: {
          id: { type: "STRING" },
          name: { type: "STRING" },
          role: { type: "STRING" },
        },
        required: ["id", "name"],
      },
    },
    utterances: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        properties: {
          speaker_id: { type: "STRING" },
          speaker_name: { type: "STRING" },
          vietnamese: { type: "STRING" },
          english: { type: "STRING" },
          intent: { type: "STRING" },
        },
        required: ["speaker_id", "speaker_name", "vietnamese", "english"],
      },
    },
  },
  required: ["utterances"],
};

async function transcribeWithGemini(filePath: string): Promise<GeminiResponse> {
  const fileData = await Bun.file(filePath).arrayBuffer();
  const base64Audio = Buffer.from(fileData).toString("base64");

  const body = {
    contents: [
      {
        parts: [
          {
            inlineData: {
              mimeType: "audio/m4a",
              data: base64Audio,
            },
          },
          {
            text: "Transcribe this Vietnamese family dinner conversation. Return structured JSON with speaker identification, Vietnamese text, and English translations for each utterance.",
          },
        ],
      },
    ],
    systemInstruction: {
      parts: [{ text: GEMINI_SYSTEM_PROMPT }],
    },
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema,
      temperature: 0.1,
      maxOutputTokens: 65536,
    },
  };

  const res = await fetch(GEMINI_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini ${res.status}: ${err}`);
  }

  const data = (await res.json()) as any;
  const candidate = data.candidates?.[0];
  const text = candidate?.content?.parts?.[0]?.text;

  if (!text) {
    throw new Error("Gemini returned no text content");
  }

  // Check for truncation
  const finishReason = candidate?.finishReason;
  if (finishReason === "MAX_TOKENS") {
    console.warn(`  \x1b[33m⚠️  Response truncated (MAX_TOKENS). Trying to salvage...\x1b[0m`);
    // Try to close the JSON array and object to make it parseable
    let fixed = text;
    // Find the last complete utterance object (ends with })
    const lastBrace = fixed.lastIndexOf("}");
    if (lastBrace > 0) {
      fixed = fixed.substring(0, lastBrace + 1) + "]}";
    }
    try {
      return JSON.parse(fixed) as GeminiResponse;
    } catch {
      throw new Error("Response was truncated and could not be repaired");
    }
  }

  try {
    return JSON.parse(text) as GeminiResponse;
  } catch {
    // Response may be truncated without MAX_TOKENS flag — try to salvage
    console.warn(`  \x1b[33m⚠️  JSON parse failed (finishReason: ${finishReason}). Attempting repair...\x1b[0m`);
    let fixed = text;
    const lastBrace = fixed.lastIndexOf("}");
    if (lastBrace > 0) {
      fixed = fixed.substring(0, lastBrace + 1) + "]}";
    }
    try {
      const parsed = JSON.parse(fixed) as GeminiResponse;
      console.warn(`  \x1b[33m⚠️  Recovered ${parsed.utterances?.length || 0} utterances from partial response\x1b[0m`);
      return parsed;
    } catch (e2: any) {
      console.error(`  \x1b[31mFirst 500 chars of response:\x1b[0m`);
      console.error(`  ${text.slice(0, 500)}`);
      throw new Error(`Failed to parse Gemini JSON: ${e2.message}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Audio splitting for large files
// ---------------------------------------------------------------------------

async function splitAudio(filePath: string): Promise<string[]> {
  const tmpDir = join(tmpdir(), `viet-listener-${Date.now()}`);
  mkdirSync(tmpDir, { recursive: true });

  const result = Bun.spawnSync([
    "ffmpeg", "-i", filePath,
    "-f", "segment",
    "-segment_time", String(CHUNK_DURATION_SEC),
    "-c", "copy",
    "-reset_timestamps", "1",
    join(tmpDir, "chunk_%03d.m4a"),
  ]);

  if (result.exitCode !== 0) {
    throw new Error(`ffmpeg split failed: ${result.stderr.toString()}`);
  }

  const chunks = readdirSync(tmpDir)
    .filter((f) => f.endsWith(".m4a"))
    .sort()
    .map((f) => join(tmpDir, f));

  return chunks;
}

function cleanupChunks(chunkPaths: string[]): void {
  for (const p of chunkPaths) {
    try { unlinkSync(p); } catch {}
  }
  // Try to remove the temp directory
  const dir = join(chunkPaths[0], "..");
  try { Bun.spawnSync(["rmdir", dir]); } catch {}
}

// ---------------------------------------------------------------------------
// Process a single clip (with auto-splitting for large files)
// ---------------------------------------------------------------------------

async function processClip(filePath: string): Promise<ProcessedUtterance[]> {
  const filename = basename(filePath);
  const clipTime = fileTimeLabel(filename);
  const fileSizeMB = (await Bun.file(filePath).arrayBuffer()).byteLength / (1024 * 1024);

  console.log(`  \x1b[36m🎙️  Processing: ${filename}\x1b[0m (${fileSizeMB.toFixed(1)}MB)`);

  // Split large files into chunks
  if (fileSizeMB > MAX_FILE_SIZE_MB) {
    console.log(`  \x1b[33m📎 File exceeds ${MAX_FILE_SIZE_MB}MB — splitting into ${CHUNK_DURATION_SEC / 60}-min chunks\x1b[0m`);

    const chunks = await splitAudio(filePath);
    console.log(`  \x1b[2m   Split into ${chunks.length} chunks\x1b[0m`);

    const allProcessed: ProcessedUtterance[] = [];

    for (let c = 0; c < chunks.length; c++) {
      console.log(`  \x1b[2m   Chunk ${c + 1}/${chunks.length}...\x1b[0m`);
      try {
        const geminiResult = await transcribeWithGemini(chunks[c]);
        if (geminiResult.utterances && geminiResult.utterances.length > 0) {
          console.log(`  \x1b[32m   ✓\x1b[0m  ${geminiResult.utterances.length} utterances`);
          const processed = geminiResult.utterances.map((u) => ({
            source_file: filename,
            clip_time: clipTime,
            speaker_name: u.speaker_name,
            vietnamese: u.vietnamese,
            english: u.english,
            intent: u.intent || undefined,
          }));
          allProcessed.push(...processed);
        }
      } catch (err: any) {
        console.error(`  \x1b[31m   ✗ Chunk ${c + 1} failed: ${err.message}\x1b[0m`);
      }
    }

    cleanupChunks(chunks);
    console.log(`  \x1b[32m✓\x1b[0m  ${allProcessed.length} total utterances from ${chunks.length} chunks`);
    return allProcessed;
  }

  // Normal processing for small files
  const geminiResult = await transcribeWithGemini(filePath);

  if (!geminiResult.utterances || geminiResult.utterances.length === 0) {
    console.log(`  \x1b[33m⚠️  No utterances detected in ${filename}\x1b[0m`);
    return [];
  }

  console.log(`  \x1b[32m✓\x1b[0m  ${geminiResult.utterances.length} utterances from Gemini`);

  if (geminiResult.session_summary) {
    console.log(`  \x1b[2m📝 ${geminiResult.session_summary}\x1b[0m`);
  }

  const processed: ProcessedUtterance[] = geminiResult.utterances.map((u) => ({
    source_file: filename,
    clip_time: clipTime,
    speaker_name: u.speaker_name,
    vietnamese: u.vietnamese,
    english: u.english,
    intent: u.intent || undefined,
  }));

  return processed;
}

// ---------------------------------------------------------------------------
// Vocabulary extraction (code-based, using dictionary)
// ---------------------------------------------------------------------------

function extractVocabulary(
  utterances: ProcessedUtterance[]
): Map<string, { count: number; translation: string }> {
  const wordCounts = new Map<string, { count: number; translation: string }>();

  for (const u of utterances) {
    const words = segmentVietnamese(u.vietnamese);
    for (const word of words) {
      const lower = word.toLowerCase();
      const translation = dictionary[lower] || dictionary[word] || "";
      const existing = wordCounts.get(lower);
      if (existing) {
        existing.count++;
      } else {
        wordCounts.set(lower, { count: 1, translation });
      }
    }
  }

  return wordCounts;
}

// ---------------------------------------------------------------------------
// Word frequency merge
// ---------------------------------------------------------------------------

async function mergeWordFrequency(
  newCounts: Map<string, { count: number; translation: string }>
): Promise<void> {
  const freqFile = Bun.file(join(scriptDir, WORD_FREQ_PATH));
  let existing: Array<{ word: string; count: number }> = [];

  if (await freqFile.exists()) {
    existing = await freqFile.json();
  }

  // Build lookup from existing
  const merged = new Map<string, number>();
  for (const entry of existing) {
    merged.set(entry.word, entry.count);
  }

  // Merge new counts
  for (const [word, { count }] of newCounts) {
    merged.set(word, (merged.get(word) || 0) + count);
  }

  // Sort by count descending
  const sorted = Array.from(merged.entries())
    .map(([word, count]) => ({ word, count }))
    .sort((a, b) => b.count - a.count);

  await Bun.write(join(scriptDir, WORD_FREQ_PATH), JSON.stringify(sorted, null, 2));
}

// ---------------------------------------------------------------------------
// Study sheet generation
// ---------------------------------------------------------------------------

function generateStudySheet(
  wordCounts: Map<string, { count: number; translation: string }>,
  date: string
): string {
  // Sort by count, take top 30
  const sorted = Array.from(wordCounts.entries())
    .filter(([_, v]) => v.translation) // only words with dictionary translations
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 30);

  let md = `# Vietnamese Study Sheet — ${date}\n\n`;
  md += `> Top ${sorted.length} words by frequency from dinner conversation\n\n`;
  md += `| # | Vietnamese | English | Count |\n`;
  md += `|---|-----------|---------|-------|\n`;

  sorted.forEach(([word, { count, translation }], i) => {
    md += `| ${i + 1} | ${word} | ${translation} | ${count} |\n`;
  });

  return md;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const target = process.argv[2];
  if (!target) {
    console.error("Usage: bun ProcessAudio.ts <folder-or-file>");
    process.exit(1);
  }

  // Preflight
  if (!GEMINI_API_KEY) {
    console.error("  ✗ GEMINI_API_KEY not set");
    process.exit(1);
  }

  // Determine files to process
  let files: string[];
  const targetFile = Bun.file(target);

  if (target.endsWith(".m4a")) {
    files = [target];
  } else {
    files = readdirSync(target)
      .filter((f) => f.endsWith(".m4a"))
      .sort()
      .map((f) => join(target, f));
  }

  if (files.length === 0) {
    console.error("  No M4A files found in", target);
    process.exit(1);
  }

  console.log(`\n  \x1b[1m🇻🇳 Processing ${files.length} Audio Clips with Gemini\x1b[0m`);
  console.log(`  \x1b[2mPipeline: Gemini 2.5 Flash → Structured JSON → Markdown + JSONL + Vocab\x1b[0m`);
  console.log(`  \x1b[2mDictionary: ${Object.keys(dictionary).length} entries loaded\x1b[0m\n`);

  // Ensure output directory exists
  const outputDir = join(scriptDir, OUTPUT_DIR);
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  const today = new Date().toISOString().slice(0, 10);
  const outputJsonl = join(outputDir, `${today}.jsonl`);
  const outputMd = join(outputDir, `${today}-transcript.md`);
  const outputVocab = join(outputDir, `${today}-vocab.md`);

  const allUtterances: ProcessedUtterance[] = [];

  // Process each clip sequentially (rate limits, visible progress)
  for (let i = 0; i < files.length; i++) {
    console.log(`\n  [${ i + 1}/${files.length}]`);
    try {
      const result = await processClip(files[i]);
      allUtterances.push(...result);
    } catch (err: any) {
      console.error(`  \x1b[31m✗ Error processing ${basename(files[i])}: ${err.message}\x1b[0m`);
    }
  }

  if (allUtterances.length === 0) {
    console.error("\n  No utterances extracted from any file.");
    process.exit(1);
  }

  // --- Write JSONL ---
  const jsonlLines = allUtterances.map((u) => JSON.stringify(u)).join("\n") + "\n";
  await Bun.write(outputJsonl, jsonlLines);
  console.log(`\n  \x1b[32m📊 JSONL:\x1b[0m ${outputJsonl}`);

  // --- Write markdown transcript ---
  let md = `---
date: ${today}
tool: ProcessAudio (Gemini)
engine: gemini-2.5-flash
language: vi → en
clips: ${files.length}
total_utterances: ${allUtterances.length}
---

# Dinner Transcript — ${today}

> ${files.length} audio clips | ${allUtterances.length} utterances
> Engine: Gemini 2.5 Flash (Vietnamese transcription + translation)

`;

  let currentFile = "";
  for (const u of allUtterances) {
    if (u.source_file !== currentFile) {
      currentFile = u.source_file;
      md += `\n## Clip: ${u.source_file} (${u.clip_time})\n\n`;
    }
    md += `**${u.speaker_name}:**\n`;
    md += `- 🇻🇳 ${u.vietnamese}\n`;
    md += `- 🇬🇧 ${u.english}\n`;
    if (u.intent) md += `- 💡 _${u.intent}_\n`;
    md += `\n`;
  }

  await Bun.write(outputMd, md);
  console.log(`  \x1b[32m📄 Transcript:\x1b[0m ${outputMd}`);

  // --- Extract vocabulary ---
  console.log(`  \x1b[2mExtracting vocabulary...\x1b[0m`);
  const wordCounts = extractVocabulary(allUtterances);

  // --- Update word-frequency.json ---
  await mergeWordFrequency(wordCounts);
  console.log(`  \x1b[35m📈 Word frequency updated:\x1b[0m ${join(scriptDir, WORD_FREQ_PATH)}`);

  // --- Generate study sheet ---
  const studySheet = generateStudySheet(wordCounts, today);
  await Bun.write(outputVocab, studySheet);
  console.log(`  \x1b[35m📚 Study sheet:\x1b[0m ${outputVocab}`);

  // --- Summary ---
  console.log(`\n  \x1b[1m✅ Done!\x1b[0m`);
  console.log(`  \x1b[2m${files.length} clips → ${allUtterances.length} utterances → JSONL + transcript + vocab\x1b[0m\n`);
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(`\n  \x1b[31mFatal: ${err.message}\x1b[0m`);
    process.exit(1);
  });
}
