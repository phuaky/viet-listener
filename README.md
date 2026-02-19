# Viet Listener

**Learning Vietnamese the way it's actually spoken — not the Duolingo way.**

I've used Duolingo for years. I still can't hold a conversation with my family in Vietnamese. The problem isn't effort — it's that Duolingo teaches you to say "the cat is on the table" while your family is saying "con lấy cho bố cái kia đi" (grab that thing for dad).

This project takes a different approach: **record real family conversations, figure out which words come up the most, and learn those first.** Not textbook Vietnamese. The Vietnamese my family actually speaks.

## The idea

1. **Record** family dinners, hangouts, normal life in Vietnam
2. **Transcribe** the audio with AI (Gemini 2.5 Flash) — get Vietnamese + English + speaker identification
3. **Analyze** word frequency — what are the 100 words that make up 80% of conversation?
4. **Learn those words** — build confidence by understanding what's actually being said around you
5. **Speak them back** — go from passive listening to active participation

The goal isn't fluency. It's **confidence**. Understanding enough to follow along, and knowing enough words to jump in.

## What's here

### Batch Processor (`ProcessAudio.ts`)

Records family conversations on your phone (M4A files), then processes them:

```
Audio files → Gemini 2.5 Flash → Structured JSON → Transcripts + Word Analysis
```

- Sends audio to Gemini with family context (who's who) for speaker identification
- Auto-splits large files (>20MB) into 10-min chunks via ffmpeg
- Outputs: JSONL (structured utterances), markdown transcript (bilingual), vocabulary study sheet
- Extracts word frequency using dictionary-based Vietnamese word segmentation

```bash
# Process a single recording
bun run ProcessAudio.ts path/to/recording.m4a

# Process a whole folder
bun run ProcessAudio.ts path/to/audio/
```

### Live Translator (`server.ts` + `index.html`)

A web app for real-time use — open it on your phone while family is talking:

- **Live Vietnamese transcription** via browser's Web Speech API
- **English translation** with ~1-2 second lag (Gemini 2.5 Flash or Google Translate)
- **Word-by-word breakdown** using a 1,057-entry Vietnamese-English dictionary
- Smart compound word detection (longest-match segmentation)

```bash
bun run dev
# → http://localhost:8765
```

Use ngrok for phone access: `ngrok http 8765`

### Dictionary (`dictionary.json`)

1,057-entry Vietnamese-English dictionary focused on everyday conversational words. Used for:
- Word segmentation (detecting compound words like "cái này" vs individual syllables)
- Instant word-level translations without API calls
- Vocabulary extraction from transcripts

## Setup

```bash
# Install
bun install

# Add your API key
echo "GEMINI_API_KEY=your-key-here" > .env

# Optional: Google Translate for faster live translation
# echo "GOOGLE_TRANSLATE_API_KEY=your-key" >> .env
```

Requires [Bun](https://bun.sh) and `ffmpeg` (for splitting large audio files).

## Early results

From ~3 hours of family dinner recordings (7 audio clips):
- **2,064 unique utterances** transcribed with speaker identification
- **1,014 Vietnamese words** identified after filtering English noise
- **317 words** appear 5+ times — these are the core family vocabulary

Top 10 most frequent words in actual family conversation:

| # | Word | Count | Meaning |
|---|------|-------|---------|
| 1 | la | 431 | is, am |
| 2 | no | 373 | he/she/it (informal) |
| 3 | khong | 311 | no, not |
| 4 | cai | 303 | classifier (for objects) |
| 5 | nay | 296 | this |
| 6 | ma | 213 | but, that |
| 7 | con | 179 | child |
| 8 | me | 178 | mom |
| 9 | cai nay | 175 | this one |
| 10 | day | 174 | there |

None of these are in Duolingo's first 50 lessons. All of them come up every 30 seconds in a real conversation.

## Why not Duolingo

Duolingo optimizes for streak retention. This optimizes for **understanding your family**.

| | Duolingo | This |
|---|---------|------|
| Vocabulary source | Textbook frequency lists | Your actual family conversations |
| First words learned | Colors, animals, "the boy eats bread" | Particles, family terms, imperatives |
| Practice method | Multiple choice, typing | Listening to real speech, real-time translation |
| Success metric | XP, streaks | "I understood what grandma said" |
| Personalization | None | 100% — your family, your conversations |

## Architecture

```
Phone (record M4A) → ProcessAudio.ts → Gemini 2.5 Flash
                                         ↓
                                    Structured JSON
                                         ↓
                              ┌──────────┼──────────┐
                              ↓          ↓          ↓
                           JSONL    Transcript    Vocab
                        (utterances) (bilingual)  (study sheet)
                                         ↓
                                   Word Frequency
                                    Analysis
```

Live mode:
```
Microphone → Web Speech API (vi-VN) → server.ts → Gemini/Google Translate
                                         ↓
                                    Word breakdown
                                    (dictionary.json)
```

## License

MIT
