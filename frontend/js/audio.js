/**
 * audio.js — Phase 3: Mic capture + WebSocket + AudioContext + Client-Side VAD.
 *
 * New in Phase 3:
 *  - Web Audio API AnalyserNode for real-time energy-based VAD
 *  - speech_start / speech_end events sent to server
 *  - Barge-in: when server sends stop_audio, drain queue and close AudioContext
 *  - Visual feedback during barge-in (UI pulse)
 */

// ── State ─────────────────────────────────────────────────────────────────────
let ws = null;
let mediaRecorder = null;
let audioContext = null;
let audioQueue = [];
let isPlaying = false;
let callActive = false;

// VAD state
let analyser = null;
let vadTimer = null;
let isSpeaking = false;         // Client-side speaking state

// VAD thresholds
const VAD_ENERGY_THRESHOLD = 0.005;   // RMS amplitude threshold (0–1) lowered for quieter mics
const VAD_SPEECH_START_MS  = 80;      // Must exceed threshold for this long → speech_start
const VAD_SPEECH_END_MS    = 700;     // Below threshold for this long → speech_end

let speechStartTimer = null;
let speechEndTimer = null;

// Latency tracking
let userStartedSpeakingAt = null;
let agentStartedSpeakingAt = null;

let pingInterval = null;

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWebSocket() {
  const wsUrl = `ws://${window.location.host}/ws/call`;
  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log("[WS] Connected");
    startMicrophone();
    
    // Stability: Start keepalive ping to prevent timeouts
    pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 10000);
  };

  ws.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer) {
      // Binary → TTS audio
      enqueueAudio(event.data);
    } else {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "pong") return; // Ignore keepalive responses
        handleServerMessage(msg);
      } catch (e) {
        console.warn("[WS] Non-JSON:", event.data);
      }
    }
  };

  ws.onerror = (err) => {
    console.error("[WS] Error:", err);
    UI.setStatus("error", "Connection error. Please refresh.");
  };

  ws.onclose = () => {
    console.log("[WS] Disconnected");
    if (pingInterval) clearInterval(pingInterval);
    if (callActive) endCall();
  };
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "status":
      onStatusChange(msg.value);
      break;

    case "transcript":
      UI.addTranscript(msg.text, msg.is_final, msg.latency_ms);
      if (msg.is_final && msg.latency_ms) {
        document.getElementById("stt-latency").textContent = Math.round(msg.latency_ms);
      }
      break;

    case "agent_text":
      UI.addAgentMessage(msg.text);
      break;

    case "stop_audio":
      // Barge-in: stop playback immediately
      stopAudioPlayback();
      UI.showBargeinPulse();
      break;

    case "interruption_ack":
      console.log("[VAD] Barge-in acknowledged by server");
      break;

    case "error":
      console.error("[Server]", msg.message);
      UI.setStatus("error", msg.message);
      break;
  }
}

function onStatusChange(value) {
  UI.setStatus(value);
  if (value === "speaking") {
    agentStartedSpeakingAt = performance.now();
    if (userStartedSpeakingAt) {
      const e2e = agentStartedSpeakingAt - userStartedSpeakingAt;
      document.getElementById("e2e-latency").textContent = Math.round(e2e);
      userStartedSpeakingAt = null;
    }
  }
}

// ── Microphone + VAD ─────────────────────────────────────────────────────────
async function startMicrophone() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    ensureAudioContext();

    // ── Client-side VAD via AnalyserNode ──────────────────────────────────
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.4;
    source.connect(analyser);
    startVADLoop();

    // ── MediaRecorder for audio streaming ─────────────────────────────────
    const mimeType = getSupportedMimeType();
    mediaRecorder = new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 16000 });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0 && ws?.readyState === WebSocket.OPEN) {
        ws.send(event.data);
      }
    };

    mediaRecorder.start(250);
    console.log("[Mic] Recording with", mimeType);
  } catch (err) {
    console.error("[Mic]", err);
    UI.setStatus("error", "Mic access denied");
    endCall();
  }
}

let dynamicThreshold = 0.005;

// ── VAD Loop ──────────────────────────────────────────────────────────────────
function startVADLoop() {
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Float32Array(bufferLength);

  let isCalibrating = true;
  let calibrationSamples = [];
  
  console.log("[VAD] Calibrating microphone for 1 second...");
  UI.setStatus("thinking", "Calibrating mic...");
  
  // Calibrate for 1 second to find the ambient noise floor
  setTimeout(() => {
    isCalibrating = false;
    if (calibrationSamples.length > 0) {
      const avgNoise = calibrationSamples.reduce((a, b) => a + b) / calibrationSamples.length;
      dynamicThreshold = Math.max(0.002, avgNoise * 2.5); // 2.5x the noise floor, min 0.002
      console.log(`[VAD] Calibration complete. Noise floor: ${avgNoise.toFixed(5)}, Threshold set to: ${dynamicThreshold.toFixed(5)}`);
      UI.setStatus("listening");
    } else {
      console.warn("[VAD] Calibration failed: 0 samples. Using default threshold.");
      UI.setStatus("listening");
    }
  }, 1000);

  function tick() {
    if (!callActive || !analyser) return;

    analyser.getFloatTimeDomainData(dataArray);

    // Compute RMS energy
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / bufferLength);
    
    if (isCalibrating) {
      if (rms > 0) calibrationSamples.push(rms);
      vadTimer = requestAnimationFrame(tick);
      return;
    }

    const isLoud = rms > dynamicThreshold;
    
    // Debug: Log RMS every half second to see if mic is working at all
    if (Math.random() < 0.03 && rms > 0) {
        console.log(`[VAD Debug] Current mic volume (RMS): ${rms.toFixed(5)}`);
    } else if (Math.random() < 0.01 && rms === 0) {
        console.log(`[VAD Debug] Mic is completely silent (RMS = 0). Check hardware mute or Windows privacy settings.`);
    }

    if (isLoud && !isSpeaking) {
      // Clear any pending end timer
      if (speechEndTimer) { clearTimeout(speechEndTimer); speechEndTimer = null; }

      // Set start timer — only fire speech_start after sustained noise
      if (!speechStartTimer) {
        speechStartTimer = setTimeout(() => {
          speechStartTimer = null;
          isSpeaking = true;
          userStartedSpeakingAt = performance.now();
          sendControl({ type: "speech_start" });
          UI.setVADIndicator(true);
        }, VAD_SPEECH_START_MS);
      }
    } else if (!isLoud && isSpeaking) {
      // Clear start timer
      if (speechStartTimer) { clearTimeout(speechStartTimer); speechStartTimer = null; }

      // Set end timer — only fire speech_end after sustained silence
      if (!speechEndTimer) {
        speechEndTimer = setTimeout(() => {
          speechEndTimer = null;
          isSpeaking = false;
          sendControl({ type: "speech_end" });
          UI.setVADIndicator(false);
        }, VAD_SPEECH_END_MS);
      }
    }

    vadTimer = requestAnimationFrame(tick);
  }

  vadTimer = requestAnimationFrame(tick);
}

function stopVAD() {
  if (vadTimer) { cancelAnimationFrame(vadTimer); vadTimer = null; }
  if (speechStartTimer) { clearTimeout(speechStartTimer); speechStartTimer = null; }
  if (speechEndTimer) { clearTimeout(speechEndTimer); speechEndTimer = null; }
  isSpeaking = false;
  analyser = null;
}

function sendControl(obj) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function stopMicrophone() {
  stopVAD();
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    mediaRecorder = null;
  }
}

function getSupportedMimeType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg"];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

// ── Audio Playback Queue ──────────────────────────────────────────────────────
function ensureAudioContext() {
  if (!audioContext || audioContext.state === "closed") {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioContext.state === "suspended") audioContext.resume();
}

function enqueueAudio(arrayBuffer) {
  audioQueue.push(arrayBuffer);
  if (!isPlaying) playNextChunk();
}

async function playNextChunk() {
  if (audioQueue.length === 0) { isPlaying = false; return; }
  isPlaying = true;
  ensureAudioContext();

  const buffer = audioQueue.shift();
  try {
    const audioBuffer = await audioContext.decodeAudioData(buffer.slice(0));
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.onended = () => playNextChunk();
    source.start(0);
  } catch (err) {
    console.warn("[Audio] Decode skip:", err.message);
    playNextChunk();
  }
}

function stopAudioPlayback() {
  audioQueue = [];
  isPlaying = false;
  if (audioContext && audioContext.state !== "closed") {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
}

// ── Call Lifecycle ────────────────────────────────────────────────────────────
function startCall() {
  callActive = true;
  UI.setCallActive(true);
  document.getElementById("latency-strip").style.display = "flex";
  ensureAudioContext();
  connectWebSocket();
}

function endCall() {
  callActive = false;
  stopMicrophone();
  stopAudioPlayback();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "end_call" }));
    ws.close();
  }
  ws = null;
  UI.setCallActive(false);
  UI.setStatus("idle", "Call ended");
}

function toggleCall() {
  callActive ? endCall() : startCall();
}

window.toggleCall = toggleCall;
