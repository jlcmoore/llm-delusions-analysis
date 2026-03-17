import {
  ANNOTATIONS_CSV_URL,
  applyDatasetLabelOrFallback,
  buildDataFileUrl,
  escapeHtml,
  installViewerNav,
  parseAnnotationCsv,
  populateGroupedAnnotationOptions,
  resolveAnnotationSpec,
  renderAnnotationSummaryHtml,
  renderContextBlock,
  renderJsonOrMarkdown,
  renderMessageContent,
} from "./viewer_shared.js";

(function () {
  "use strict";

  const elements = {
    layout: document.querySelector(".manual-layout"),
    sidebar: document.querySelector(".manual-sidebar"),
    instructionsSection: document.getElementById("instructions-section"),
    runSelect: document.getElementById("run-select"),
    fileInput: document.getElementById("dataset-input"),
    loaderStatus: document.getElementById("loader-status"),
    annotationStatus: document.getElementById("annotation-status"),
    statusLine: document.getElementById("status-line"),
    annotationSelect: document.getElementById("annotation-select"),
    participantSelect: document.getElementById("participant-select"),
    showContextToggle: document.getElementById("show-context"),
    annotationDetails: document.getElementById("annotation-details"),
    instructionsBody: document.getElementById("instructions-body"),
    navigationStatus: document.getElementById("navigation-status"),
    participantBanner: document.getElementById("participant-banner"),
    messageCard: document.getElementById("message-card"),
    messageMeta: document.getElementById("message-meta"),
    messageContent: document.getElementById("message-content"),
    precedingContext: document.getElementById("preceding-context"),
    showRawContentToggle: document.getElementById("show-raw-content"),
    btnYes: document.getElementById("btn-yes"),
    btnNo: document.getElementById("btn-no"),
    btnBack: document.getElementById("btn-back"),
    btnToggleContext: document.getElementById("btn-toggle-context"),
    btnTogglePanes: document.getElementById("btn-toggle-panes"),
    btnDownload: document.getElementById("btn-download"),
    btnNextUnlabeled: document.getElementById("btn-next-unlabeled"),
    annotatorInput: document.getElementById("annotator-id"),
    resumeStatus: document.getElementById("resume-status"),
  };

  const state = {
    records: [],
    participants: [],
    annotationSpecs: [],
    annotationById: {},
    annotationLoaded: false,
    currentAnnotation: "__none__",
    currentParticipant: "__all__",
    showContext: true,
    showPanes: true,
    showRawContent: false,
    filteredRecords: [],
    currentIndex: 0,
    labels: new Map(),
    datasetSourcePath: "",
    serverBacked: false,
    annotatorId: "",
    pendingServerLabelIds: new Set(),
  };

  function setInstructions(sharedText) {
    const container = elements.instructionsBody;
    if (!container) {
      return;
    }
    const base = (sharedText || "").trim()
      ? escapeHtml(sharedText)
          .replace(/\r\n/g, "\n")
          .replace(/\r/g, "\n")
          .replace(/\n\n+/g, "</p><p>")
          .replace(/\n/g, "<br>")
      : escapeHtml(
          "You will read one target message at a time and decide whether it satisfies the selected annotation.",
        );
    container.innerHTML = `
      <p>${base}</p>
      <p><strong>Keyboard controls:</strong></p>
      <ul>
        <li><strong>Yes</strong> — press the button or key <code>Y</code>.</li>
        <li><strong>No</strong> — press the button or key <code>N</code>.</li>
        <li><strong>Back</strong> — return to the previous message with the button or the <code>Space</code> key.</li>
        <li><strong>Navigate without changing labels</strong> — move between messages with the <code>&larr;</code> (previous) and <code>&rarr;</code> (next) arrow keys.</li>
        <li><strong>Next unlabeled</strong> — jump to the next unlabeled message with the <code>U</code> key.</li>
        <li><strong>Toggle context</strong> — show or hide preceding context with the <code>P</code> key.</li>
        <li><strong>Toggle original text</strong> — show or hide the unformatted message text with the <code>O</code> key.</li>
        <li><strong>Toggle instructions</strong> — show or hide the sidebar and details panes with the <code>I</code> key.</li>
      </ul>
      <p>
        Focus on the target message when deciding. Use preceding context only when it helps interpret meaning.
        Do not highlight or quote spans; simply choose Yes or No.
      </p>
    `;
  }

  async function loadHumanInstructions() {
    try {
      const response = await fetch("/api/manual-instructions", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      const text =
        data && typeof data.instructions === "string" ? data.instructions : "";
      setInstructions(text);
    } catch (error) {
      console.error("Failed to load manual instructions:", error);
      setInstructions("");
    }
  }

  async function getJsonlParser() {
    const response = await fetch(
      "../../node_modules/jsonl-parse-stringify/index.js",
      { cache: "force-cache" },
    );
    const source = await response.text();
    const shim = { exports: {} };
    const load = new Function("exports", "module", source);
    load(shim.exports, shim);
    return shim.exports.default || shim.exports;
  }

  function setLoaderStatus(message, isError) {
    if (!elements.loaderStatus) {
      return;
    }
    elements.loaderStatus.textContent = message;
    if (isError) {
      elements.loaderStatus.classList.add("error");
    } else {
      elements.loaderStatus.classList.remove("error");
    }
  }

  function setAnnotationStatus(message, isError) {
    if (!elements.annotationStatus) {
      return;
    }
    elements.annotationStatus.textContent = message;
    if (isError) {
      elements.annotationStatus.classList.add("error");
    } else {
      elements.annotationStatus.classList.remove("error");
    }
  }

  async function parseJsonl(text, sourceLabel) {
    const parser = await getJsonlParser();
    try {
      const results = parser.parse(text);
      return results.map((data) => ({ data, sourceLabel }));
    } catch (error) {
      throw new Error(
        `Failed to parse ${sourceLabel}: ${error && error.message ? error.message : error}`,
      );
    }
  }

  function readJsonlFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        parseJsonl(String(reader.result || ""), file.name)
          .then(resolve)
          .catch(reject);
      };
      reader.onerror = () => {
        reject(reader.error || new Error(`Unable to read ${file.name}`));
      };
      reader.readAsText(file);
    });
  }

  function normalizeRecord(raw, sourceLabel, indexInFile) {
    if (!raw) {
      return null;
    }
    const type = raw.type || "item";
    if (type === "meta") {
      return null;
    }
    const annotationId = raw.annotation_id ? String(raw.annotation_id) : "";
    const sequenceIndex =
      typeof raw.sequence_index === "number" &&
      Number.isFinite(raw.sequence_index)
        ? raw.sequence_index
        : indexInFile;
    const participant = raw.participant ? String(raw.participant) : "";
    const id = [
      sourceLabel || "",
      annotationId || "unknown",
      String(sequenceIndex),
      participant,
    ]
      .filter(Boolean)
      .join("|");
    const preceding = Array.isArray(raw.preceding) ? raw.preceding : [];
    return {
      id,
      source: sourceLabel || "",
      sequenceIndex,
      participant,
      annotationId,
      annotationLabel: raw.annotation
        ? String(raw.annotation)
        : annotationId || "",
      chatKey: raw.chat_key || "",
      chatDate: raw.chat_date || null,
      chatIndex:
        typeof raw.chat_index === "number" && Number.isFinite(raw.chat_index)
          ? raw.chat_index
          : null,
      messageIndex:
        typeof raw.message_index === "number" &&
        Number.isFinite(raw.message_index)
          ? raw.message_index
          : null,
      role: raw.role || "",
      timestamp: raw.timestamp || null,
      content: raw.content || "",
      preceding: preceding.map((msg) => ({
        index:
          typeof msg.index === "number" && Number.isFinite(msg.index)
            ? msg.index
            : null,
        role: msg.role || "",
        content: msg.content || "",
        timestamp: msg.timestamp || null,
      })),
    };
  }

  function hasDataset() {
    return Array.isArray(state.records) && state.records.length > 0;
  }

  function updateStatusLine() {
    if (!elements.statusLine) {
      return;
    }
    if (!hasDataset()) {
      elements.statusLine.textContent =
        "Load a manual annotation dataset to begin.";
      return;
    }
    const total = state.filteredRecords.length;
    if (!total || state.currentAnnotation === "__none__") {
      elements.statusLine.textContent =
        "Select an annotation to start annotating messages.";
      return;
    }
    const idx = state.currentIndex + 1;
    const labeledCount = state.filteredRecords.filter((record) =>
      state.labels.has(record.id),
    ).length;
    elements.statusLine.textContent =
      `Annotating ${escapeHtml(state.currentAnnotation)} ` +
      `\u2022 Message ${idx} of ${total} ` +
      `\u2022 Labeled ${labeledCount} of ${total}`;
  }

  function updateParticipantOptions() {
    const select = elements.participantSelect;
    if (!select) {
      return;
    }
    while (select.options.length > 1) {
      select.remove(1);
    }
    if (!hasDataset()) {
      select.disabled = true;
      select.value = "__all__";
      return;
    }
    const participants = Array.from(
      new Set(
        state.records.map((record) => record.participant).filter(Boolean),
      ),
    ).sort();
    state.participants = participants;
    participants.forEach((participant) => {
      const option = document.createElement("option");
      option.value = participant;
      option.textContent = participant;
      select.appendChild(option);
    });
    select.disabled = false;
    if (
      state.currentParticipant !== "__all__" &&
      !participants.includes(state.currentParticipant)
    ) {
      state.currentParticipant = "__all__";
    }
    select.value = state.currentParticipant;
  }

  function getAnnotationSpec(annotationId) {
    if (!annotationId || annotationId === "__none__") {
      return null;
    }
    return resolveAnnotationSpec(
      annotationId,
      state.annotationSpecs,
      state.annotationById,
    );
  }

  function updateAnnotationOptions() {
    const select = elements.annotationSelect;
    if (!select) {
      return;
    }
    while (select.options.length > 1) {
      select.remove(1);
    }
    if (!hasDataset()) {
      select.disabled = true;
      select.value = "__none__";
      return;
    }
    const annotationIds = Array.from(
      new Set(
        state.records.map((record) => record.annotationId).filter(Boolean),
      ),
    );
    // Group annotations by category using the shared helper so the dropdown
    // matches the classification and agreement viewers.
    state.currentAnnotation = populateGroupedAnnotationOptions(
      select,
      annotationIds,
      state.annotationSpecs,
      {},
      state.currentAnnotation,
      "__none__",
    );
    select.disabled = false;
    if (
      state.currentAnnotation === "__none__" ||
      !annotationIds.includes(state.currentAnnotation)
    ) {
      state.currentAnnotation = annotationIds[0] || "__none__";
    }
    select.value = state.currentAnnotation;
  }

  function updateAnnotationDetails() {
    const container = elements.annotationDetails;
    if (!container) {
      return;
    }
    const spec = getAnnotationSpec(state.currentAnnotation);
    if (!spec) {
      container.hidden = true;
      container.innerHTML = "";
      return;
    }
    container.hidden = false;
    container.innerHTML = renderAnnotationSummaryHtml(spec, {
      showExamples: true,
    });
  }

  function filterRecords() {
    if (!hasDataset()) {
      state.filteredRecords = [];
      return;
    }
    const records = state.records.filter((record) => {
      if (!record.annotationId) {
        return false;
      }
      if (
        state.currentAnnotation !== "__none__" &&
        record.annotationId !== state.currentAnnotation
      ) {
        return false;
      }
      if (
        state.currentParticipant !== "__all__" &&
        record.participant !== state.currentParticipant
      ) {
        return false;
      }
      return true;
    });
    records.sort((a, b) => {
      const aSeq = typeof a.sequenceIndex === "number" ? a.sequenceIndex : 0;
      const bSeq = typeof b.sequenceIndex === "number" ? b.sequenceIndex : 0;
      if (aSeq !== bSeq) {
        return aSeq - bSeq;
      }
      if (a.participant !== b.participant) {
        return a.participant < b.participant ? -1 : 1;
      }
      if (a.chatKey !== b.chatKey) {
        return a.chatKey < b.chatKey ? -1 : 1;
      }
      const aIdx =
        typeof a.messageIndex === "number"
          ? a.messageIndex
          : Number.MAX_SAFE_INTEGER;
      const bIdx =
        typeof b.messageIndex === "number"
          ? b.messageIndex
          : Number.MAX_SAFE_INTEGER;
      return aIdx - bIdx;
    });
    state.filteredRecords = records;
    let firstUnlabeled = records.findIndex(
      (record) => !state.labels.has(record.id),
    );
    if (firstUnlabeled === -1) {
      firstUnlabeled = 0;
    }
    state.currentIndex = firstUnlabeled;
  }

  function updateNavigationStatus() {
    if (!elements.navigationStatus) {
      return;
    }
    if (!hasDataset() || !state.filteredRecords.length) {
      elements.navigationStatus.textContent = "No messages available.";
      return;
    }
    const total = state.filteredRecords.length;
    const idx = state.currentIndex + 1;
    const record = state.filteredRecords[state.currentIndex];
    const metaParts = [];
    if (record.participant) {
      metaParts.push(`Participant: ${escapeHtml(record.participant)}`);
    }
    if (record.chatKey) {
      metaParts.push(`Chat: ${escapeHtml(record.chatKey)}`);
    }
    if (record.chatDate) {
      metaParts.push(`Date: ${escapeHtml(record.chatDate)}`);
    }
    const labelMeta = state.labels.get(record.id);
    let labelChipClass = "label-chip label-missing";
    let labelText = "label: unlabeled";
    if (labelMeta) {
      if (labelMeta.label === "yes") {
        labelChipClass = "label-chip label-yes";
        labelText = "label: yes";
      } else if (labelMeta.label === "no") {
        labelChipClass = "label-chip label-no";
        labelText = "label: no";
      }
    }
    const metaHtml = metaParts.length
      ? ` \u2022 ${metaParts.join(" \u2022 ")}`
      : "";
    elements.navigationStatus.innerHTML =
      `Message ${idx} of ${total}${metaHtml} \u2022 ` +
      `<span class="${labelChipClass}">${escapeHtml(labelText)}</span>`;
  }

  function renderCurrentMessage() {
    const card = elements.messageCard;
    const meta = elements.messageMeta;
    const body = elements.messageContent;
    const ctxBefore = elements.precedingContext;
    if (!card || !meta || !body || !ctxBefore) {
      return;
    }
    if (!hasDataset() || !state.filteredRecords.length) {
      card.hidden = true;
      meta.textContent = "";
      body.textContent = "";
      ctxBefore.innerHTML = "";
      ctxBefore.hidden = true;
      updateNavigationStatus();
      updateStatusLine();
      return;
    }
    const index = Math.max(
      0,
      Math.min(state.currentIndex, state.filteredRecords.length - 1),
    );
    const record = state.filteredRecords[index];
    const metaParts = [];
    if (record.role) {
      const roleLabel =
        record.role.charAt(0).toUpperCase() +
        record.role.slice(1).toLowerCase();
      metaParts.push(roleLabel);
    }
    if (record.timestamp) {
      metaParts.push(String(record.timestamp));
    }
    meta.textContent = metaParts.length ? metaParts.join(" • ") : "";
    body.innerHTML = renderMessageContent(record.content, state.showRawContent);

    if (state.showContext && Array.isArray(record.preceding)) {
      const ctxHtml = renderContextBlock(
        "Preceding context",
        record.preceding,
        "before",
        renderJsonOrMarkdown,
      );
      ctxBefore.innerHTML = ctxHtml;
      ctxBefore.hidden = false;
    } else {
      ctxBefore.innerHTML = "";
      ctxBefore.hidden = true;
    }

    const banner = elements.participantBanner;
    if (banner) {
      const prev = index > 0 ? state.filteredRecords[index - 1] || null : null;
      const changed = !prev || prev.participant !== record.participant;
      if (changed && record.participant) {
        banner.textContent = `Now annotating participant ${record.participant}`;
        banner.hidden = false;
      } else {
        banner.hidden = true;
        banner.textContent = "";
      }
    }

    card.hidden = false;
    updateNavigationStatus();
    updateStatusLine();
    updateButtonStates(record);
  }

  function updateButtonStates(record) {
    const yes = elements.btnYes;
    const no = elements.btnNo;
    if (!yes || !no) {
      return;
    }
    const label = state.labels.get(record.id);
    yes.classList.remove("matching");
    no.classList.remove("non-matching");
    if (!label) {
      return;
    }
    if (label.label === "yes") {
      yes.classList.add("matching");
      no.classList.remove("non-matching");
    } else if (label.label === "no") {
      no.classList.add("non-matching");
      yes.classList.remove("matching");
    }
  }

  function applySelection(labelValue) {
    if (!hasDataset() || !state.filteredRecords.length) {
      return;
    }
    const record = state.filteredRecords[state.currentIndex];
    const now = new Date().toISOString();
    state.labels.set(record.id, {
      label: labelValue,
      at: now,
      annotationId: record.annotationId,
      participant: record.participant,
    });
    if (elements.btnDownload) {
      elements.btnDownload.disabled = state.labels.size === 0;
    }
    if (state.serverBacked && state.datasetSourcePath) {
      state.pendingServerLabelIds.add(record.id);
      // Fire and forget; errors are surfaced via alert/status.
      void saveLabelsToServer();
    }
    if (state.currentIndex < state.filteredRecords.length - 1) {
      state.currentIndex += 1;
    }
    renderCurrentMessage();
    saveLabelsAutoIfFinished();
  }

  function goBack() {
    if (!hasDataset() || !state.filteredRecords.length) {
      return;
    }
    if (state.currentIndex > 0) {
      state.currentIndex -= 1;
    }
    renderCurrentMessage();
  }

  function goForward() {
    if (!hasDataset() || !state.filteredRecords.length) {
      return;
    }
    if (state.currentIndex < state.filteredRecords.length - 1) {
      state.currentIndex += 1;
    }
    renderCurrentMessage();
  }

  function goToNextUnlabeled() {
    if (!hasDataset() || !state.filteredRecords.length) {
      return;
    }
    const total = state.filteredRecords.length;
    if (total === 0) {
      return;
    }
    const startIndex = state.currentIndex;
    let index = startIndex;
    for (let offset = 0; offset < total; offset += 1) {
      index = (startIndex + offset + 1) % total;
      const record = state.filteredRecords[index];
      if (record && !state.labels.has(record.id)) {
        state.currentIndex = index;
        renderCurrentMessage();
        return;
      }
    }
    if (elements.navigationStatus) {
      const container = elements.navigationStatus;
      let pill = container.querySelector(".all-labeled-indicator");
      if (!pill) {
        pill = document.createElement("span");
        pill.className = "label-chip label-missing all-labeled-indicator";
        if (container.childNodes.length) {
          container.appendChild(document.createTextNode(" "));
        }
        container.appendChild(pill);
      }
      pill.textContent = "No unlabeled messages remain.";
    }
  }

  function handleToggleContext() {
    state.showContext = !state.showContext;
    if (elements.showContextToggle) {
      elements.showContextToggle.checked = state.showContext;
    }
    if (elements.btnToggleContext) {
      elements.btnToggleContext.textContent = state.showContext
        ? "Hide context (P)"
        : "Show context (P)";
    }
    renderCurrentMessage();
  }

  function applyPaneVisibility() {
    const sidebar = elements.sidebar;
    const instructionsSection = elements.instructionsSection;
    const annotationDetails = elements.annotationDetails;
    const show = state.showPanes;
    if (sidebar) {
      sidebar.style.display = show ? "" : "none";
    }
    if (instructionsSection) {
      instructionsSection.hidden = !show;
    }
    if (annotationDetails) {
      if (!show) {
        annotationDetails.hidden = true;
      } else {
        updateAnnotationDetails();
      }
    }
    if (elements.btnTogglePanes) {
      elements.btnTogglePanes.textContent = show
        ? "Hide instructions (I)"
        : "Show instructions (I)";
    }
  }

  function handleTogglePanes() {
    state.showPanes = !state.showPanes;
    applyPaneVisibility();
  }

  function buildLabelRecords(idsFilter) {
    const records = [];
    state.records.forEach((record) => {
      const meta = state.labels.get(record.id);
      if (!meta) {
        return;
      }
      if (idsFilter && !idsFilter.has(record.id)) {
        return;
      }
      records.push({
        id: record.id,
        participant: record.participant,
        annotation_id: record.annotationId,
        annotation: record.annotationLabel,
        source: record.source,
        chat_key: record.chatKey,
        chat_date: record.chatDate,
        chat_index: record.chatIndex,
        message_index: record.messageIndex,
        role: record.role,
        timestamp: record.timestamp,
        content: record.content,
        preceding: record.preceding,
        sequence_index: record.sequenceIndex,
        label: meta.label,
        labeled_at: meta.at,
      });
    });
    return records;
  }

  function handleDownloadLabels() {
    if (!state.labels.size || !hasDataset()) {
      return;
    }
    const payloads = buildLabelRecords();
    if (!payloads.length) {
      return;
    }
    const lines = payloads.map((item) => JSON.stringify(item));
    const blob = new Blob([lines.join("\n") + "\n"], {
      type: "application/x-ndjson",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = `manual_annotations_${timestamp}.jsonl`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function saveLabelsToServer() {
    if (!state.serverBacked || !state.datasetSourcePath) {
      return;
    }
    const annotatorRaw = state.annotatorId ? state.annotatorId.trim() : "";
    if (!annotatorRaw) {
      alert("Enter an annotator identifier before saving.");
      return;
    }
    const idsFilter = state.pendingServerLabelIds;
    if (!idsFilter || !idsFilter.size) {
      return;
    }
    const payloads = buildLabelRecords(idsFilter);
    if (!payloads.length) {
      return;
    }
    try {
      const response = await fetch("/api/save-manual-labels", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_path: state.datasetSourcePath,
          annotator_id: annotatorRaw,
          labels: payloads,
        }),
      });
      let data;
      try {
        data = await response.json();
      } catch {
        data = null;
      }
      if (!response.ok || !data || data.error) {
        const message =
          (data && data.error) || `HTTP ${response.status} while saving labels`;
        throw new Error(message);
      }
      const target = data.path || state.datasetSourcePath;
      setLoaderStatus(`Saved ${payloads.length} labels to ${target}.`, false);
      state.pendingServerLabelIds.clear();
    } catch (error) {
      console.error("Failed to save labels to server:", error);
      alert(
        `Failed to save labels on server: ${
          error && error.message ? error.message : error
        }`,
      );
    }
  }

  async function saveLabelsAutoIfFinished() {
    if (!hasDataset() || !state.filteredRecords.length) {
      return;
    }
    const total = state.filteredRecords.length;
    const isLast = state.currentIndex >= total - 1;
    if (!isLast) {
      return;
    }
    const allLabeled = state.filteredRecords.every((record) =>
      state.labels.has(record.id),
    );
    if (!allLabeled) {
      return;
    }
    const confirmed = window.confirm(
      "You have labeled all messages for this annotation and participant filter. Save labels now?",
    );
    if (!confirmed) {
      return;
    }
    if (state.serverBacked && state.datasetSourcePath) {
      await saveLabelsToServer();
    } else {
      handleDownloadLabels();
    }
  }

  async function loadAnnotationSpecs() {
    try {
      const response = await fetch(ANNOTATIONS_CSV_URL, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const text = await response.text();
      state.annotationSpecs = parseAnnotationCsv(text).filter(
        (spec) => spec.id,
      );
      state.annotationById = Object.fromEntries(
        state.annotationSpecs.map((spec) => [spec.id, spec]),
      );
      setAnnotationStatus(
        `Loaded ${state.annotationSpecs.length} annotation definitions.`,
        false,
      );
      state.annotationLoaded = true;
      if (elements.fileInput) {
        elements.fileInput.disabled = false;
      }
    } catch (error) {
      state.annotationSpecs = [];
      state.annotationById = {};
      state.annotationLoaded = false;
      const runningFromFile =
        typeof window !== "undefined" &&
        window.location &&
        window.location.protocol === "file:";
      const message = runningFromFile
        ? "Start a local HTTP server so annotations.csv can be fetched."
        : "Unable to load annotations.csv automatically. Ensure the server is running and annotations.csv is reachable.";
      setAnnotationStatus(message, true);
      if (elements.fileInput) {
        elements.fileInput.disabled = true;
      }
      setLoaderStatus(
        "Viewer disabled: annotation metadata failed to load.",
        true,
      );

      console.error(error);
    }
  }

  async function handleFileSelection(event) {
    if (!state.annotationLoaded) {
      setLoaderStatus(
        "Annotation metadata not loaded; start the local server before uploading records.",
        true,
      );
      if (event && event.target) {
        event.target.value = "";
      }
      return;
    }
    const files = Array.from((event.target && event.target.files) || []);
    if (!files.length) {
      setLoaderStatus("No files selected.", true);
      return;
    }
    setLoaderStatus("Loading manual annotation records…", false);
    try {
      const fileRecords = await Promise.all(files.map(readJsonlFile));
      const combined = fileRecords.flat();
      setDatasetFromParsedRecords(
        combined,
        `Loaded ${combined.length} raw records from ${files.length} file(s).`,
        null,
      );
    } catch (error) {
      console.error(error);
      setLoaderStatus(
        error && error.message
          ? error.message
          : "Failed to load manual annotation records.",
        true,
      );
    }
  }

  function setDatasetFromParsedRecords(
    parsedRecords,
    successMessage,
    sourcePath,
  ) {
    const combined = Array.isArray(parsedRecords) ? parsedRecords : [];
    const normalized = [];
    combined.forEach((record, index) => {
      const value = normalizeRecord(record.data, record.sourceLabel, index);
      if (value) {
        normalized.push(value);
      }
    });
    if (!normalized.length) {
      setLoaderStatus(
        "No usable item records found in the selected dataset.",
        true,
      );
      state.records = [];
      state.filteredRecords = [];
      renderCurrentMessage();
      updateParticipantOptions();
      updateAnnotationOptions();
      updateAnnotationDetails();
      return;
    }
    state.records = normalized;
    state.labels.clear();
    state.datasetSourcePath = sourcePath || "";
    state.serverBacked = Boolean(sourcePath);
    // Restore last-used annotation and participant when possible so that
    // refreshes return annotators to the same view.
    try {
      if (window.localStorage) {
        const storedAnnotation = window.localStorage.getItem(
          "manualCurrentAnnotationId",
        );
        if (storedAnnotation) {
          const hasAnnotation = normalized.some(
            (record) => record.annotationId === storedAnnotation,
          );
          if (hasAnnotation) {
            state.currentAnnotation = storedAnnotation;
          }
        }
        const storedParticipant = window.localStorage.getItem(
          "manualCurrentParticipantId",
        );
        if (storedParticipant) {
          const hasParticipant = normalized.some(
            (record) => record.participant === storedParticipant,
          );
          if (hasParticipant) {
            state.currentParticipant = storedParticipant;
          }
        }
      }
    } catch {
      // Ignore storage errors; fall back to defaults.
    }
    if (elements.btnDownload) {
      elements.btnDownload.disabled = true;
    }
    filterRecords();
    updateParticipantOptions();
    updateAnnotationOptions();
    updateAnnotationDetails();
    renderCurrentMessage();
    if (successMessage) {
      setLoaderStatus(successMessage, false);
    }
  }

  async function loadDatasetFromUrl(relativePath) {
    if (!relativePath) {
      return;
    }
    const url = buildDataFileUrl(relativePath);
    if (!url) {
      setLoaderStatus("Invalid dataset path.", true);
      return;
    }
    setLoaderStatus(`Loading ${relativePath}…`, false);
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const text = await response.text();
      const parsed = await parseJsonl(text, relativePath);
      setDatasetFromParsedRecords(
        parsed,
        `Loaded ${parsed.length} records from ${relativePath}.`,
        relativePath,
      );
      await resumeLabelsFromServerIfAvailable();
    } catch (error) {
      console.error("Failed to load dataset from URL:", error);
      setLoaderStatus(
        error && error.message
          ? error.message
          : "Failed to load dataset from server.",
        true,
      );
    }
  }

  async function loadAvailableDatasets() {
    const select = elements.runSelect;
    if (!select) {
      return;
    }
    select.disabled = true;
    select.innerHTML =
      '<option value="">Scanning manual_annotation_inputs/…</option>';
    try {
      const response = await fetch("/api/manual-datasets", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      const items = Array.isArray(data.datasets) ? data.datasets : [];
      select.innerHTML =
        '<option value="">Select a dataset from manual_annotation_inputs/</option>';
      if (!items.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent =
          "No datasets found; generate one with prepare_manual_annotation_dataset.py";
        select.appendChild(option);
        select.disabled = true;
        return;
      }
      items.forEach((item) => {
        if (!item || !item.path) {
          return;
        }
        const option = document.createElement("option");
        option.value = item.path;
        option.textContent = applyDatasetLabelOrFallback(item, item.path);
        select.appendChild(option);
      });
      select.disabled = false;
      // When possible, restore the last-used dataset selection so refreshes
      // resume from the same source.
      try {
        if (window.localStorage) {
          const lastPath = window.localStorage.getItem("manualDatasetPath");
          if (lastPath) {
            const matching = Array.from(select.options).find(
              (opt) => opt.value === lastPath,
            );
            if (matching) {
              select.value = lastPath;
              // Trigger a load without firing a synthetic change event.
              await loadDatasetFromUrl(lastPath);
            }
          }
        }
      } catch {
        // Ignore storage errors; users can still pick a dataset manually.
      }
    } catch (error) {
      console.error("Failed to load manual datasets:", error);
      select.innerHTML =
        '<option value="">Unable to scan manual_annotation_inputs/</option>';
      select.disabled = true;
    }
  }

  async function resumeLabelsFromServerIfAvailable() {
    if (
      !state.serverBacked ||
      !state.datasetSourcePath ||
      !state.annotatorId ||
      !hasDataset()
    ) {
      return;
    }
    const params = new URLSearchParams({
      dataset_path: state.datasetSourcePath,
      annotator_id: state.annotatorId,
    });
    try {
      const response = await fetch(`/api/manual-labels?${params.toString()}`, {
        cache: "no-store",
      });
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      const items = Array.isArray(data.labels) ? data.labels : [];
      if (!items.length) {
        return;
      }
      const recordById = new Map();
      state.records.forEach((record) => {
        recordById.set(record.id, record);
      });
      items.forEach((item) => {
        if (!item || !item.id) {
          return;
        }
        const record = recordById.get(String(item.id));
        if (!record) {
          return;
        }
        const labelRaw = typeof item.label === "string" ? item.label : "";
        const labelValue =
          labelRaw === "no" || labelRaw === "yes" ? labelRaw : null;
        if (!labelValue) {
          return;
        }
        const at =
          typeof item.labeled_at === "string" && item.labeled_at.trim()
            ? item.labeled_at
            : new Date().toISOString();
        state.labels.set(record.id, {
          label: labelValue,
          at,
          annotationId: record.annotationId,
          participant: record.participant,
        });
      });
      // Recompute filtered view and jump to first unlabeled.
      filterRecords();
      renderCurrentMessage();
      if (elements.resumeStatus) {
        elements.resumeStatus.textContent =
          items.length === 1
            ? "Resumed 1 label from a previous session."
            : `Resumed ${items.length} labels from a previous session.`;
        elements.resumeStatus.hidden = false;
      }
    } catch (error) {
      console.error("Failed to resume labels from server:", error);
    }
  }

  function attachHandlers() {
    if (elements.annotatorInput) {
      try {
        const cached =
          window.localStorage &&
          window.localStorage.getItem("manualAnnotatorId");
        if (cached) {
          state.annotatorId = cached;
          elements.annotatorInput.value = cached;
        }
      } catch {
        // ignore storage errors
      }
      elements.annotatorInput.addEventListener("input", async (event) => {
        const previousId = state.annotatorId || "";
        const value = String(event.target.value || "").trim();
        state.annotatorId = value;
        try {
          if (window.localStorage) {
            window.localStorage.setItem("manualAnnotatorId", value);
          }
        } catch {
          // ignore storage errors
        }
        if (value !== previousId) {
          state.labels.clear();
          if (elements.btnDownload) {
            elements.btnDownload.disabled = true;
          }
          filterRecords();
          renderCurrentMessage();
          if (
            state.serverBacked &&
            state.datasetSourcePath &&
            state.annotatorId
          ) {
            await resumeLabelsFromServerIfAvailable();
          }
        }
      });
    }
    if (elements.runSelect) {
      elements.runSelect.addEventListener("change", (event) => {
        const value = event.target.value;
        if (!value) {
          return;
        }
        try {
          if (window.localStorage) {
            window.localStorage.setItem("manualDatasetPath", value);
          }
        } catch {
          // ignore storage errors
        }
        loadDatasetFromUrl(value);
      });
    }
    if (elements.fileInput) {
      elements.fileInput.addEventListener("change", handleFileSelection);
    }
    if (elements.annotationSelect) {
      elements.annotationSelect.addEventListener("change", (event) => {
        state.currentAnnotation = event.target.value;
        try {
          if (window.localStorage) {
            window.localStorage.setItem(
              "manualCurrentAnnotationId",
              state.currentAnnotation,
            );
          }
        } catch {
          // ignore storage errors
        }
        filterRecords();
        updateAnnotationDetails();
        applyPaneVisibility();
        renderCurrentMessage();
      });
    }
    if (elements.participantSelect) {
      elements.participantSelect.addEventListener("change", (event) => {
        state.currentParticipant = event.target.value;
        try {
          if (window.localStorage) {
            window.localStorage.setItem(
              "manualCurrentParticipantId",
              state.currentParticipant,
            );
          }
        } catch {
          // ignore storage errors
        }
        filterRecords();
        applyPaneVisibility();
        renderCurrentMessage();
      });
    }
    if (elements.showContextToggle) {
      elements.showContextToggle.addEventListener("change", (event) => {
        state.showContext = Boolean(event.target.checked);
        if (elements.btnToggleContext) {
          elements.btnToggleContext.textContent = state.showContext
            ? "Hide context (P)"
            : "Show context (P)";
        }
        renderCurrentMessage();
      });
    }
    if (elements.showRawContentToggle) {
      elements.showRawContentToggle.addEventListener("change", (event) => {
        state.showRawContent = Boolean(event.target.checked);
        renderCurrentMessage();
      });
    }
    if (elements.btnYes) {
      elements.btnYes.addEventListener("click", () => applySelection("yes"));
    }
    if (elements.btnNo) {
      elements.btnNo.addEventListener("click", () => applySelection("no"));
    }
    if (elements.btnBack) {
      elements.btnBack.addEventListener("click", () => goBack());
    }
    if (elements.btnToggleContext) {
      elements.btnToggleContext.addEventListener("click", () =>
        handleToggleContext(),
      );
    }
    if (elements.btnTogglePanes) {
      elements.btnTogglePanes.addEventListener("click", () =>
        handleTogglePanes(),
      );
    }
    if (elements.btnDownload) {
      elements.btnDownload.addEventListener("click", () =>
        handleDownloadLabels(),
      );
    }
    if (elements.btnNextUnlabeled) {
      elements.btnNextUnlabeled.addEventListener("click", () =>
        goToNextUnlabeled(),
      );
    }
    document.addEventListener("keydown", (event) => {
      const target = event.target;
      const isInput =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable);
      if (isInput) {
        return;
      }
      if (event.key === "y" || event.key === "Y") {
        event.preventDefault();
        applySelection("yes");
        return;
      }
      if (event.key === "n" || event.key === "N") {
        event.preventDefault();
        applySelection("no");
        return;
      }
      if (event.key === "p" || event.key === "P") {
        event.preventDefault();
        handleToggleContext();
        return;
      }
      if (event.key === "o" || event.key === "O") {
        event.preventDefault();
        state.showRawContent = !state.showRawContent;
        if (elements.showRawContentToggle) {
          elements.showRawContentToggle.checked = state.showRawContent;
        }
        renderCurrentMessage();
        return;
      }
      if (event.key === "i" || event.key === "I") {
        event.preventDefault();
        handleTogglePanes();
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        goForward();
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        goBack();
        return;
      }
      if (event.key === "u" || event.key === "U") {
        event.preventDefault();
        goToNextUnlabeled();
        return;
      }
      if (event.code === "Space" || event.key === " ") {
        event.preventDefault();
        goBack();
      }
    });
  }

  function initialize() {
    installViewerNav("manual");
    attachHandlers();
    updateStatusLine();
    renderCurrentMessage();
    loadAnnotationSpecs();
    loadHumanInstructions();
    loadAvailableDatasets();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
