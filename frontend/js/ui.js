const UI = (() => {
  const avatarEl = document.getElementById("avatar");
  const avatarRing = document.getElementById("avatar-ring");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const statusBadge = document.getElementById("status-badge");
  const transcriptFeed = document.getElementById("transcript-feed");
  const transcriptContainer = document.getElementById("transcript-container");
  const suggestions = document.getElementById("suggestions");
  const btnCall = document.getElementById("btn-call");
  const aiHero = document.getElementById("ai-hero");
  const bargeinOverlay = document.getElementById("bargein-overlay");

  let currentInterimEl = null;
  let currentAgentEl = null;
  let isFirstMessage = true;

  function setStatus(value, customText = null) {
    statusBadge.className = "status-pill";
    avatarEl.classList.remove("speaking");
    avatarRing.classList.remove("active");

    const stateMap = {
      listening: { class: "listening", text: "Listening...", ring: "active" },
      thinking: { class: "thinking", text: "Thinking...", ring: "" },
      speaking: { class: "speaking", text: "Speaking", ring: "active" },
      idle: { class: "idle", text: "Ready", ring: "" },
      error: { class: "error", text: "Error", ring: "" }
    };

    const state = stateMap[value] || { class: "", text: value, ring: "" };
    
    statusBadge.classList.add(state.class);
    statusText.textContent = customText || state.text;

    if (state.ring === "active") avatarRing.classList.add("active");
    if (value === "speaking") avatarEl.classList.add("speaking");
  }

  function setCallActive(active) {
    if (active) {
      btnCall.classList.add("active");
      suggestions.classList.add("hidden");
      transcriptContainer.style.display = "flex";
      aiHero.classList.add("compact");
      // Change mic icon to stop icon
      document.getElementById("btn-icon").innerHTML = '<path d="M6 6h12v12H6z"/>';
    } else {
      btnCall.classList.remove("active");
      setStatus("idle");
      avatarRing.classList.remove("active");
      avatarEl.classList.remove("speaking");
      suggestions.classList.remove("hidden");
      transcriptContainer.style.display = "none";
      aiHero.classList.remove("compact");
      // Change back to mic icon
      document.getElementById("btn-icon").innerHTML = '<path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>';
      currentAgentEl = null;
    }
  }

  function clearPlaceholder() {
    if (isFirstMessage) {
      transcriptFeed.innerHTML = "";
      isFirstMessage = false;
    }
  }

  function addTranscript(text, isFinal, latencyMs) {
    clearPlaceholder();
    if (!isFinal) {
      if (!currentInterimEl) {
        currentInterimEl = createMessageEl("user");
        currentInterimEl.style.opacity = "0.7";
        transcriptFeed.appendChild(currentInterimEl);
      }
      currentInterimEl.textContent = text;
      scrollToBottom();
      return;
    }
    if (currentInterimEl) {
      currentInterimEl.remove();
      currentInterimEl = null;
    }
    const el = createMessageEl("user");
    el.textContent = text;
    transcriptFeed.appendChild(el);
    currentAgentEl = null;
    scrollToBottom();
  }

  function addAgentMessage(text) {
    clearPlaceholder();
    if (!currentAgentEl) {
      currentAgentEl = createMessageEl("agent");
      transcriptFeed.appendChild(currentAgentEl);
    }
    const existing = currentAgentEl.textContent;
    currentAgentEl.textContent = existing ? existing + " " + text : text;
    scrollToBottom();
  }

  function createMessageEl(role) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    return el;
  }

  function scrollToBottom() {
    transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
  }

  function setVADIndicator(isSpeaking) {
    if (isSpeaking) {
      btnCall.classList.add("vad-active");
    } else {
      btnCall.classList.remove("vad-active");
    }
  }

  function showBargeinPulse() {
    bargeinOverlay.classList.remove("bargein-anim");
    void bargeinOverlay.offsetWidth; // trigger reflow
    bargeinOverlay.classList.add("bargein-anim");
    statusText.textContent = "Interrupted ✂";
    
    if (currentAgentEl) {
      currentAgentEl.classList.add("interrupted");
    }
    
    setTimeout(() => {
      if (statusBadge.classList.contains("listening")) {
        statusText.textContent = "Listening...";
      }
    }, 1000);
  }

  return { setStatus, setCallActive, addTranscript, addAgentMessage, setVADIndicator, showBargeinPulse };
})();

window.UI = UI;

window.startCallWithHint = function(hintText) {
  if (!window.callActive) {
    window.toggleCall();
  }
}
