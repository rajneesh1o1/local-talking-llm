# Visual Context Integration

## Overview

This system integrates real-time visual observation with the LLM chat system, making the AI context-aware of its surroundings.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│  Vision Observer    │         │   Chat System        │
│  (observe.py)       │─────────│   (chat_tts.py)      │
│                     │  JSON   │                      │
│  - Object Detection │  File   │  - Voice Input       │
│  - Face Recognition │ Storage │  - LLM Processing    │
│  - Activity Detect  │         │  - TTS Output        │
└─────────────────────┘         └──────────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                 visual_context.json
                 (Last 5 observations)
```

## Components

### 1. Vision Observer (`vision/observe.py`)

**What it does:**
- Runs continuously in the background
- Detects objects using YOLO every 1 second
- Recognizes faces from `images/` folder
- Tracks facial expressions (smiling, talking, head direction)
- Monitors hand gestures (open, fist, fingers extended)
- Saves last 5 detections to `visual_context.json`

**Output Format:**
```json
{
  "visual_history": [
    {
      "timestamp": "2026-01-23T16:23:47",
      "detections": [
        {
          "object": "person",
          "person_data": {
            "name": "rajneesh",
            "face": {
              "smiling": true,
              "talking": true,
              "head_direction": "center"
            },
            "left_hand": null,
            "right_hand": {
              "present": true,
              "open": true,
              "fist": false
            }
          },
          "confidence": 0.92,
          "bbox": [185, 190, 427, 511]
        },
        {
          "object": "cell phone",
          "confidence": 0.34,
          "bbox": [195, 231, 235, 326]
        }
      ]
    }
  ]
}
```

### 2. Chat System Integration (`chat_tts.py`)

**New Function: `get_visual_context(last_n=3)`**
- Reads `visual_context.json`
- Retrieves last N observations (default 3)
- Formats visual data for LLM consumption
- Returns human-readable context string

**LLM Prompt Enhancement:**
The visual context is now prepended to every LLM query:

```
=== VISUAL CONTEXT (What I can see around you) ===

Timestamp: 2026-01-23T16:23:47
Detected 2 object(s):
  - Person: rajneesh (smiling, talking, looking center), right hand open
  - cell phone (confidence: 0.34)

=== END VISUAL CONTEXT ===

[Memory Context from PostgreSQL]

User: [User's spoken message]
```

### 3. Updated System Prompt (`llm/chat.py`)

**Key Additions:**
- Explains vision capabilities to the LLM
- Instructs LLM to use visual context naturally
- Encourages contextually aware responses
- Enables LLM to reference what it "sees"

**Significance:**
```
VISUAL CONTEXT SIGNIFICANCE:
- You receive real-time visual observations
- Includes: objects, people, expressions, gestures
- Use visual info for contextually aware responses
- Acknowledge user's actions (smiling, gesturing)
- Reference visible objects when relevant
- Personalize responses using recognized names
```

## Usage

### Option 1: Run Systems Together (Recommended)

```bash
python start_vision_chat.py
```

This launcher will:
1. Start vision observer in background thread
2. Start chat system in main thread
3. Both systems communicate via `visual_context.json`
4. Press Ctrl+C to stop everything

### Option 2: Run Systems Separately

**Terminal 1 - Vision Observer:**
```bash
cd vision
python observe.py
```

**Terminal 2 - Chat System:**
```bash
python chat_tts.py
```

## Storage Details

**File:** `visual_context.json`
**Location:** Project root directory
**Max Entries:** 5 (automatically maintains last 5 detections)
**Thread-Safe:** Uses threading locks for concurrent access
**Update Frequency:** Every 1 second (detection_interval)

## Visual Context in LLM Calls

**When user speaks:**
1. Voice → Text conversion
2. Load last 3 visual observations from JSON
3. Load relevant memories from PostgreSQL
4. Combine: Visual Context + Memory Context + User Message
5. Send to LLM for context-aware response
6. LLM can reference what it sees in its reply

**Example LLM Response:**
```
User: "How do I look?"

LLM Response (with visual context):
"I can see you're smiling right now# which looks great!# 
Your expression seems happy and engaged# That's wonderful to see!"
```

## Benefits

1. **Spatial Awareness**: LLM knows what objects are around the user
2. **Emotional Intelligence**: Detects and responds to facial expressions
3. **Gesture Recognition**: Understands hand gestures and body language
4. **Personalization**: Recognizes users by face and uses their names
5. **Natural Interaction**: Responses feel more present and observant
6. **Context Retention**: Maintains short-term visual memory (last 5 observations)

## Performance

**Vision Observer:**
- Detection interval: 1 second
- RAM usage: ~1.5 GB (YOLO + MediaPipe)
- CPU usage: Moderate (M4 optimized)
- No impact on chat system performance

**Chat System:**
- Reads JSON file only when user speaks
- Minimal overhead (<10ms)
- No blocking operations
- Thread-safe file access

## Limitations

1. **Single Camera**: Currently supports one camera feed
2. **Single Person Focus**: Best with one person in frame
3. **No Recording**: Visual data is temporary (not saved to PostgreSQL)
4. **Face Recognition Accuracy**: Uses histogram matching (simple but effective)
5. **Storage Limit**: Only last 5 observations kept

## Future Enhancements

- [ ] Multiple person tracking
- [ ] Object interaction detection (e.g., "user picked up phone")
- [ ] Activity sequence recognition (e.g., "user has been coding for 30 mins")
- [ ] Emotion classification from facial features
- [ ] Voice tone + visual expression correlation
- [ ] Optional visual memory storage in PostgreSQL

## Troubleshooting

**Vision window not showing:**
- Camera permissions may be blocked
- Another app might be using the camera

**No visual context in LLM:**
- Check if `visual_context.json` exists in project root
- Ensure observe.py is running
- Check file permissions

**LLM not acknowledging visual context:**
- Visual context might be empty
- Check system prompt in `llm/chat.py`
- Verify visual_context.json has data

## Security Note

Visual observations are:
- ✅ Stored locally only (`visual_context.json`)
- ✅ Never uploaded to cloud
- ✅ Automatically expire (last 5 only)
- ✅ Only text descriptions sent to LLM (no images)
- ✅ Face recognition data stays local

## Credits

Built using:
- YOLO (Ultralytics) - Object detection
- MediaPipe (Google) - Face & hand tracking
- OpenCV - Face recognition
- Threading - Concurrent execution

