/**
 * Viet Listener — Translation Backend
 *
 * A lightweight Bun server that proxies Google Cloud Translation API.
 * Provides sentence-level translation + word-level breakdown.
 *
 * Usage:
 *   GOOGLE_TRANSLATE_API_KEY=your-key bun run server.ts
 *
 * Endpoints:
 *   GET  /                → serves static files (index.html, etc.)
 *   POST /api/translate   → translates Vietnamese sentence + word breakdown
 */

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const GOOGLE_API_KEY = process.env.GOOGLE_TRANSLATE_API_KEY || '';
const API_KEY = GOOGLE_API_KEY || GEMINI_API_KEY; // either works
const PORT = 8765;
const GOOGLE_TRANSLATE_URL = 'https://translation.googleapis.com/language/translate/v2';
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;

// ─── Translation Cache ───────────────────────────────────────
const translationCache = new Map<string, string>();

async function geminiTranslate(texts: string[], source: string = 'vi', target: string = 'en'): Promise<string[]> {
  // Check cache first
  const uncachedTexts: string[] = [];
  const uncachedIndices: number[] = [];
  const results: string[] = new Array(texts.length);

  for (let i = 0; i < texts.length; i++) {
    const cacheKey = `${source}:${target}:${texts[i]}`;
    const cached = translationCache.get(cacheKey);
    if (cached) {
      results[i] = cached;
    } else {
      uncachedTexts.push(texts[i]);
      uncachedIndices.push(i);
    }
  }

  if (uncachedTexts.length === 0) return results;

  const srcLang = source === 'vi' ? 'Vietnamese' : 'English';
  const tgtLang = target === 'en' ? 'English' : 'Vietnamese';

  // Batch into groups of 20 for Gemini
  const batchSize = 20;
  for (let b = 0; b < uncachedTexts.length; b += batchSize) {
    const batch = uncachedTexts.slice(b, b + batchSize);
    const numbered = batch.map((t, i) => `${i + 1}. ${t}`).join('\n');

    const res = await fetch(GEMINI_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text:
          `Translate each line from ${srcLang} to ${tgtLang}. Return ONLY the translations, one per line, numbered to match. No explanations.\n\n${numbered}`
        }] }],
        generationConfig: { temperature: 0.1, maxOutputTokens: 4096 },
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Gemini API error: ${res.status} ${err}`);
    }

    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';

    // Parse numbered lines
    const lines = text.split('\n').filter((l: string) => l.trim());
    for (let j = 0; j < batch.length; j++) {
      let translated = lines[j]?.replace(/^\d+\.\s*/, '').trim() || batch[j];
      const globalIdx = b + j;
      const originalIdx = uncachedIndices[globalIdx];
      results[originalIdx] = translated;

      const cacheKey = `${source}:${target}:${batch[j]}`;
      translationCache.set(cacheKey, translated);
    }
  }

  return results;
}

async function googleTranslate(texts: string[], source: string = 'vi', target: string = 'en'): Promise<string[]> {
  // If no Google key, fall back to Gemini
  if (!GOOGLE_API_KEY && GEMINI_API_KEY) {
    return geminiTranslate(texts, source, target);
  }
  if (!GOOGLE_API_KEY) {
    throw new Error('No translation API key set');
  }

  const uncachedTexts: string[] = [];
  const uncachedIndices: number[] = [];
  const results: string[] = new Array(texts.length);

  for (let i = 0; i < texts.length; i++) {
    const cacheKey = `${source}:${target}:${texts[i]}`;
    const cached = translationCache.get(cacheKey);
    if (cached) {
      results[i] = cached;
    } else {
      uncachedTexts.push(texts[i]);
      uncachedIndices.push(i);
    }
  }

  if (uncachedTexts.length === 0) return results;

  const batches: string[][] = [];
  for (let i = 0; i < uncachedTexts.length; i += 128) {
    batches.push(uncachedTexts.slice(i, i + 128));
  }

  let uncachedIdx = 0;
  for (const batch of batches) {
    const res = await fetch(GOOGLE_TRANSLATE_URL, {
      method: 'POST',
      headers: {
        'X-goog-api-key': GOOGLE_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        q: batch,
        source,
        target,
        format: 'text',
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Google Translate API error: ${res.status} ${err}`);
    }

    const data = await res.json() as {
      data: { translations: { translatedText: string }[] }
    };

    for (let j = 0; j < data.data.translations.length; j++) {
      const translated = data.data.translations[j].translatedText;
      const originalIdx = uncachedIndices[uncachedIdx];
      results[originalIdx] = translated;

      const cacheKey = `${source}:${target}:${batch[j]}`;
      translationCache.set(cacheKey, translated);
      uncachedIdx++;
    }
  }

  return results;
}

// ─── Vietnamese Word Segmentation ────────────────────────────
// Load dictionary for compound word detection
const dictFile = Bun.file('./dictionary.json');
let dictionary: Record<string, string> = {};
try {
  dictionary = await dictFile.json();
  console.log(`Dictionary loaded: ${Object.keys(dictionary).length} entries`);
} catch (e) {
  console.warn('dictionary.json not found, will rely on Google Translate for all words');
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

// ─── API Handler ─────────────────────────────────────────────

interface TranslateRequest {
  text: string;
  mode?: 'vi-to-en' | 'en-to-vi';
}

interface WordBreakdown {
  word: string;
  translation: string;
  fromDictionary: boolean;
}

interface TranslateResponse {
  sentence: {
    original: string;
    translation: string;
  };
  words: WordBreakdown[];
}

async function handleTranslate(req: Request): Promise<Response> {
  const body = await req.json() as TranslateRequest;
  const { text, mode = 'vi-to-en' } = body;

  if (!text || text.trim().length === 0) {
    return Response.json({ error: 'No text provided' }, { status: 400 });
  }

  const source = mode === 'vi-to-en' ? 'vi' : 'en';
  const target = mode === 'vi-to-en' ? 'en' : 'vi';

  try {
    // Step 1: Translate the full sentence
    const [sentenceTranslation] = await googleTranslate([text], source, target);

    // Step 2: Smart word segmentation + translation (for vi-to-en)
    let words: WordBreakdown[] = [];

    if (mode === 'vi-to-en') {
      words = await smartSegmentAndTranslate(text);
    }

    const response: TranslateResponse = {
      sentence: {
        original: text,
        translation: sentenceTranslation,
      },
      words,
    };

    return Response.json(response);
  } catch (err: any) {
    console.error('Translation error:', err.message);
    return Response.json({ error: err.message }, { status: 500 });
  }
}

/**
 * Smart segmentation using Google Translate for compound detection.
 *
 * Strategy: batch-translate ALL n-grams (1, 2, 3 syllables) via Google in
 * ONE request. Then use dynamic programming to find the optimal segmentation
 * by scoring each candidate span — compounds that translate as a single
 * concept score higher than the sum of their individual syllables.
 */
async function smartSegmentAndTranslate(text: string): Promise<WordBreakdown[]> {
  const cleaned = text.trim()
    .replace(/[.,!?;:"""''()[\]{}]/g, ' ')
    .replace(/\s+/g, ' ');
  const syllables = cleaned.split(' ').filter(s => s.length > 0);

  if (syllables.length === 0) return [];

  const N = syllables.length;
  const MAX_SPAN = Math.min(4, N);

  // ─── Step 1: Collect all n-grams and their dictionary translations ───
  // spans[i][len] = { text, translation, fromDictionary, score }
  type Span = { text: string; translation: string; fromDictionary: boolean; score: number };
  const spans: (Span | null)[][] = Array.from({ length: N }, () => Array(MAX_SPAN + 1).fill(null));

  const toTranslate: { text: string; i: number; len: number }[] = [];

  for (let i = 0; i < N; i++) {
    for (let len = 1; len <= Math.min(MAX_SPAN, N - i); len++) {
      const phrase = syllables.slice(i, i + len).join(' ');
      const lower = phrase.toLowerCase();
      const dictEntry = dictionary[lower] || dictionary[phrase];
      if (dictEntry) {
        spans[i][len] = {
          text: phrase,
          translation: dictEntry,
          fromDictionary: true,
          score: len * 10, // dictionary entries get high confidence
        };
      } else {
        toTranslate.push({ text: phrase, i, len });
      }
    }
  }

  // ─── Step 2: Batch-translate all unknown n-grams via Google ───
  if (toTranslate.length > 0 && API_KEY) {
    const translations = await googleTranslate(toTranslate.map(t => t.text), 'vi', 'en');
    for (let j = 0; j < toTranslate.length; j++) {
      const { i, len, text: phrase } = toTranslate[j];
      spans[i][len] = {
        text: phrase,
        translation: translations[j],
        fromDictionary: false,
        score: 0, // scored in step 3
      };
    }
  } else if (toTranslate.length > 0) {
    // No API key — fill with the text itself
    for (const { i, len, text: phrase } of toTranslate) {
      spans[i][len] = { text: phrase, translation: phrase, fromDictionary: false, score: 0 };
    }
  }

  // ─── Step 3: Score multi-syllable spans ───
  // A span of len>1 is a compound if its translation is meaningfully different
  // from the concatenation of its individual syllable translations.
  for (let i = 0; i < N; i++) {
    for (let len = 2; len <= Math.min(MAX_SPAN, N - i); len++) {
      const span = spans[i][len];
      if (!span || span.fromDictionary) continue; // dict entries already scored

      const compoundTrans = span.translation.toLowerCase().trim();

      // Build expected "non-compound" translation by concatenating singles
      const parts: string[] = [];
      for (let k = 0; k < len; k++) {
        const single = spans[i + k]?.[1];
        if (single) {
          parts.push((single.translation.split(',')[0] || '').trim().toLowerCase());
        }
      }
      const concat = parts.join(' ');
      const concatReversed = [...parts].reverse().join(' ');

      // Compound if translation differs from naive concatenation
      const isCompound =
        compoundTrans &&
        compoundTrans !== concat &&
        compoundTrans !== concatReversed &&
        compoundTrans.split(' ').length <= 3; // single concept

      if (isCompound) {
        span.score = len * 8; // compound detected via API
        // Cache for future sessions
        dictionary[span.text.toLowerCase()] = span.translation;
      } else {
        span.score = len * 1; // not a real compound — slight preference for grouping but low
      }
    }
  }

  // Ensure all single-syllable spans have a baseline score
  for (let i = 0; i < N; i++) {
    const s = spans[i][1];
    if (s && s.score === 0) s.score = 1;
  }

  // ─── Step 4: Dynamic programming — find optimal segmentation ───
  // dp[i] = best total score for syllables[0..i-1]
  const dp: number[] = new Array(N + 1).fill(-Infinity);
  const choice: { start: number; len: number }[] = new Array(N + 1);
  dp[0] = 0;

  for (let i = 0; i < N; i++) {
    if (dp[i] === -Infinity) continue;
    for (let len = 1; len <= Math.min(MAX_SPAN, N - i); len++) {
      const span = spans[i][len];
      if (!span) continue;
      const newScore = dp[i] + span.score;
      if (newScore > dp[i + len]) {
        dp[i + len] = newScore;
        choice[i + len] = { start: i, len };
      }
    }
  }

  // ─── Step 5: Backtrack to build result ───
  const segments: { start: number; len: number }[] = [];
  let pos = N;
  while (pos > 0) {
    const c = choice[pos];
    segments.push(c);
    pos = c.start;
  }
  segments.reverse();

  const results: WordBreakdown[] = [];
  for (const { start, len } of segments) {
    const span = spans[start][len]!;
    results.push({
      word: span.text,
      translation: span.translation,
      fromDictionary: span.fromDictionary,
    });
  }

  return results;
}

// ─── Suggestion handler for Reply mode ──────────────────────

interface SuggestRequest {
  englishText: string;
  learnedWords: Record<string, number>; // word → frequency
}

async function handleSuggest(req: Request): Promise<Response> {
  const body = await req.json() as SuggestRequest;
  const { englishText, learnedWords = {} } = body;

  if (!englishText) {
    return Response.json({ error: 'No text provided' }, { status: 400 });
  }

  try {
    // Translate English to Vietnamese via Google
    const [vietnameseTranslation] = await googleTranslate([englishText], 'en', 'vi');

    // Also generate 2 alternative phrasings by translating slight variations
    const variations = [
      englishText,
      `I want to say: ${englishText}`,
      `Reply: ${englishText}`,
    ];

    const allTranslations = await googleTranslate(variations, 'en', 'vi');

    // Deduplicate
    const uniqueSuggestions = [...new Set(allTranslations)].slice(0, 3);

    // For each suggestion, check how many words overlap with learned vocabulary
    const suggestions = await Promise.all(uniqueSuggestions.map(async (vi) => {
      const words = segmentVietnamese(vi);
      let knownCount = 0;
      const wordDetails = words.map(w => {
        const lower = w.toLowerCase();
        const isKnown = !!learnedWords[lower];
        if (isKnown) knownCount++;
        return { word: w, known: isKnown, frequency: learnedWords[lower] || 0 };
      });

      // Get English back-translation for verification
      const [backTranslation] = await googleTranslate([vi], 'vi', 'en');

      return {
        vietnamese: vi,
        english: backTranslation,
        words: wordDetails,
        knownRatio: words.length > 0 ? knownCount / words.length : 0,
        knownCount,
      };
    }));

    // Sort by known word ratio (higher = more familiar vocabulary)
    suggestions.sort((a, b) => b.knownRatio - a.knownRatio);

    return Response.json({ suggestions });
  } catch (err: any) {
    console.error('Suggest error:', err.message);
    return Response.json({ error: err.message }, { status: 500 });
  }
}

// ─── Server ──────────────────────────────────────────────────

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (req.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // API routes
    if (url.pathname === '/api/translate' && req.method === 'POST') {
      const res = await handleTranslate(req);
      // Add CORS headers
      for (const [k, v] of Object.entries(corsHeaders)) {
        res.headers.set(k, v);
      }
      return res;
    }

    if (url.pathname === '/api/suggest' && req.method === 'POST') {
      const res = await handleSuggest(req);
      for (const [k, v] of Object.entries(corsHeaders)) {
        res.headers.set(k, v);
      }
      return res;
    }

    // Practice progress (server-side persistence)
    if (url.pathname === '/api/progress' && req.method === 'GET') {
      const file = Bun.file('./practice-progress.json');
      if (await file.exists()) {
        return Response.json(await file.json(), { headers: corsHeaders });
      }
      return Response.json({}, { headers: corsHeaders });
    }

    if (url.pathname === '/api/progress' && req.method === 'POST') {
      const body = await req.json();
      await Bun.write('./practice-progress.json', JSON.stringify(body, null, 2));
      return Response.json({ saved: true }, { headers: corsHeaders });
    }

    // API key check endpoint
    if (url.pathname === '/api/status') {
      return Response.json({
        hasApiKey: !!API_KEY,
        dictionarySize: Object.keys(dictionary).length,
        cacheSize: translationCache.size,
      }, { headers: corsHeaders });
    }

    // Static files
    let filePath = url.pathname === '/' ? '/index.html' : url.pathname;
    const file = Bun.file('.' + filePath);
    if (await file.exists()) {
      return new Response(file, { headers: corsHeaders });
    }

    return new Response('Not found', { status: 404 });
  },
});

if (!API_KEY) {
  console.warn('\n⚠️  No translation API key set!');
  console.warn('   Set GEMINI_API_KEY or GOOGLE_TRANSLATE_API_KEY in .env');
  console.warn('   App will still work but translations will fall back to dictionary only.\n');
}

const translationBackend = GOOGLE_API_KEY ? 'Google Translate' : GEMINI_API_KEY ? 'Gemini 2.5 Flash' : 'dictionary only';
console.log(`🇻🇳 Viet Listener server running at http://localhost:${PORT}`);
console.log(`   Translation: ${translationBackend}`);
console.log(`   Dictionary: ${Object.keys(dictionary).length} words`);
