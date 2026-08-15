/**
 * ui.js — UI state management for the call interface.
 *
 * Phase 3 additions:
 *  - setVADIndicator: mic glow when user is speaking (client-side VAD)
 *  - showBargeinPulse: flash animation when user interrupts agent
 */

const UI = (() => {
  // DOM references
  const avatarEl      = document.getElementById("avatar");
  const avatarRing    = document.getElementById("avatar-ring");
  const statusDot     = document.getElementById("status-dot");
  const statusText    = document.getElementById("status-text");
  const transcriptFeed = document.getElementById("transcript-feed");
  const btnCall       = document.getElementById("btn-call");
  const btnIcon       = document.getElementById("btn-icon");
  const btnLabel      = document.getElementById("btn-label");

  // Track the current interim message element so we can update it in-place
  let currentInterimEl = null;
  // Track the current agent message being streamed sentence by sentence
  let currentAgentEl = null;
  let isFirstMessage = true;

  // ── Status ────────────────────────────────────────────────────────────────
  function setStatus(value, customText = null) {
    // Remove all state classes
    statusDot.className = "status-dot";
    avatarEl.classList.remove("speaking");
    avatarRing.classList.remove("active", "pulsing");

    const stateMap = {
      listening: { dot: "listening", text: "Listening...",  ring: "active" },
      thinking:  { dot: "thinking",  text: "Thinking...",   ring: "pulsing" },
      speaking:  { dot: "speaking",  text: "Aria speaking", ring: "pulsing" },
      idle:      { dot: "",          text: "Call ended",    ring: "" },
      error:     { dot: "",          text: "Error",         ring: "" },
    };

    const state = stateMap[value] || { dot: "", text: value, ring: "" };

    if (state.dot) statusDot.classList.add(state.dot);
    statusText.textContent = customText || state.text;

    if (state.ring === "active")   avatarRing.classList.add("active");
    if (state.ring === "pulsing")  avatarRing.classList.add("pulsing");
    if (value === "speaking")      avatarEl.classList.add("speaking");
  }

  // ── Call Button ───────────────────────────────────────────────────────────
  function setCallActive(active) {
    if (active) {
      btnCall.classList.add("active");
      btnIcon.textContent = "📵";
      btnLabel.textContent = "End Call";
    } else {
      btnCall.classList.remove("active");
      btnIcon.textContent = "📞";
      btnLabel.textContent = "Start Call";
      setStatus("idle");
      avatarRing.classList.remove("active", "pulsing");
      avatarEl.classList.remove("speaking");
      currentAgentEl = null;
    }
  }

  // ── Transcript ────────────────────────────────────────────────────────────
  function clearPlaceholder() {
    if (isFirstMessage) {
      transcriptFeed.innerHTML = "";
      isFirstMessage = false;
    }
  }

  /**
   * Add or update the user's interim (live) transcript.
   * Interim results update in-place; final results commit a new bubble.
   */
  function addTranscript(text, isFinal, latencyMs) {
    clearPlaceholder();

    if (!isFinal) {
      // Interim: update existing interim bubble or create one
      if (!currentInterimEl) {
        currentInterimEl = createMessageEl("user");
        const bubble = currentInterimEl.querySelector(".bubble");
        bubble.classList.add("interim");
        transcriptFeed.appendChild(currentInterimEl);
      }
      currentInterimEl.querySelector(".bubble").textContent = text;
      scrollToBottom();
      return;
    }

    // Final transcript: remove interim bubble, add committed bubble
    if (currentInterimEl) {
      currentInterimEl.remove();
      currentInterimEl = null;
    }

    const el = createMessageEl("user");
    el.querySelector(".bubble").textContent = text;
    if (latencyMs) {
      el.querySelector(".bubble-label").textContent = `STT ${Math.round(latencyMs)}ms`;
    }
    transcriptFeed.appendChild(el);
    currentAgentEl = null;  // New user turn → next agent msg is fresh
    scrollToBottom();
  }

  /**
   * Add an agent message sentence. Sentences for the same turn
   * are appended to the same bubble (streaming feel).
   */
  function addAgentMessage(text) {
    clearPlaceholder();

    if (!currentAgentEl) {
      currentAgentEl = createMessageEl("agent");
      transcriptFeed.appendChild(currentAgentEl);
    }

    const bubble = currentAgentEl.querySelector(".bubble");
    const existing = bubble.textContent;
    bubble.textContent = existing ? existing + " " + text : text;
    scrollToBottom();
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function createMessageEl(role) {
    const el = document.createElement("div");
    el.className = `message ${role}`;

    const label = document.createElement("span");
    label.className = "bubble-label";
    label.textContent = role === "user" ? "You" : "Aria";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "user") {
      el.appendChild(bubble);
      el.appendChild(label);
    } else {
      el.appendChild(label);
      el.appendChild(bubble);
    }

    return el;
  }

  function scrollToBottom() {
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  }

  // ── Phase 3: VAD Indicator ────────────────────────────────────────────────
  /**
   * Show/hide a glowing mic indicator on the call button
   * when the client-side VAD detects user speech.
   */
  function setVADIndicator(isSpeaking) {
    if (isSpeaking) {
      btnCall.classList.add("user-speaking");
    } else {
      btnCall.classList.remove("user-speaking");
    }
  }

  /**
   * Flash the avatar ring briefly to signal a barge-in was detected.
   */
  function showBargeinPulse() {
    avatarRing.classList.add("bargein-flash");
    setTimeout(() => avatarRing.classList.remove("bargein-flash"), 600);
    statusText.textContent = "Interrupted ✂";
    setTimeout(() => statusText.textContent = "Listening...", 1000);
  }

  return { setStatus, setCallActive, addTranscript, addAgentMessage, setVADIndicator, showBargeinPulse };
})();

window.UI = UI;
