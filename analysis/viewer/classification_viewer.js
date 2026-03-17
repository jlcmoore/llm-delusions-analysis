import {
  ANNOTATIONS_CSV_URL,
  applyDatasetLabelOrFallback,
  clampContextDepth,
  escapeHtml,
  installViewerNav,
  parseAnnotationCsv,
  populateGroupedAnnotationOptions,
  renderTextBlock,
  resolveAnnotationSpec,
  renderContextBlock,
  renderMessageContent,
  renderMatchList,
  installScoreCutoffControl,
} from "./viewer_shared.js";

(function () {
  "use strict";

  /**
   * Cache frequently accessed DOM nodes so the event handlers stay lean.
   */
  const elements = {
    fileInput: document.getElementById("jsonl-input"),
    loaderStatus: document.getElementById("loader-status"),
    annotationStatus: document.getElementById("annotation-status"),
    summary: document.getElementById("dataset-summary"),
    datasetSelect: document.getElementById("dataset-select"),
    participantSelect: document.getElementById("participant-select"),
    annotationSelect: document.getElementById("annotation-select"),
    scopeFilterSelect: document.getElementById("scope-filter"),
    keywordFilterInput: document.getElementById("keyword-filter"),
    pageSizeSelect: document.getElementById("page-size"),
    showMatching: document.getElementById("show-matching"),
    showNonMatching: document.getElementById("show-non-matching"),
    annotationDetails: document.getElementById("annotation-details"),
    results: document.getElementById("results"),
    resultsAlert: document.getElementById("results-alert"),
    controls: document.querySelector(".controls"),
    showRawContentToggle: null,
    showErrorsToggle: null,
    onlyErrorsToggle: null,
    contextDepthSelect: null,
  };

  const state = {
    participant: "__all__",
    annotation: "__none__",
    pageSize: 25,
    page: 1,
    showMatching: true,
    showNonMatching: false,
    dataset: null,
    datasetKey: null,
    datasets: [],
    records: [],
    totalRecords: 0,
    showRawContent: false,
    contextDepth: 1,
    scoreCutoff: 5,
    scoreMode: "ge",
    cutoffsByAnnotation: {},
    scopeFilter: "__all__",
    keywordFilter: "",
  };

  let annotationSpecs = [];
  let annotationById = {};
  let annotationLoaded = false;
  let recordCacheCounter = 1;
  const recordCache = new Map();
  // Augmented metadata captured from JSONL meta lines
  // Populated by the dataset assembler when available.
  // Shape: { [annotation_id]: { name, description } }
  // Also, dataset.annotationMeta may contain { system_prompt, template }.

  const errorClassName = "error";

  const collator = new Intl.Collator("en", {
    numeric: true,
    sensitivity: "base",
  });

  let configPromise = null;

  function getConfig() {
    if (!configPromise) {
      configPromise = (async () => {
        try {
          const response = await fetch("/api/config", { cache: "no-store" });
          const data = await response.json();
          if (!response.ok || !data || data.error) {
            throw new Error(
              data && data.error ? data.error : `HTTP ${response.status}`,
            );
          }
          return data;
        } catch (error) {
          console.error("Failed to fetch viewer config:", error);
          return {};
        }
      })();
    }
    return configPromise;
  }

  /**
   * Update the file loader status line with optional error styling.
   */
  function setLoaderStatus(message, isError) {
    if (!elements.loaderStatus) {
      return;
    }
    elements.loaderStatus.textContent = message;
    if (isError) {
      elements.loaderStatus.classList.add(errorClassName);
    } else {
      elements.loaderStatus.classList.remove(errorClassName);
    }
  }

  /**
   * Update annotation metadata load status with optional error styling.
   */
  function setAnnotationStatus(message, isError) {
    if (!elements.annotationStatus) {
      return;
    }
    elements.annotationStatus.textContent = message;
    if (isError) {
      elements.annotationStatus.classList.add(errorClassName);
    } else {
      elements.annotationStatus.classList.remove(errorClassName);
    }
  }

  function normalizeLineBreaks(value) {
    return String(value || "")
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n")
      .trim();
  }

  /**
   * True once we have at least one normalized record in memory.
   */
  function hasDataset() {
    return Boolean(
      state.dataset &&
        typeof state.totalRecords === "number" &&
        state.totalRecords >= 0,
    );
  }

  /**
   * Fetch and parse annotations.csv so the viewer can display rich metadata.
   */
  async function loadAnnotationSpecs() {
    try {
      const response = await fetch(ANNOTATIONS_CSV_URL, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const text = await response.text();
      annotationSpecs = parseAnnotationCsv(text).filter((spec) => spec.id);
      annotationById = Object.fromEntries(
        annotationSpecs.map((spec) => [spec.id, spec]),
      );
      setAnnotationStatus(
        `Loaded ${annotationSpecs.length} annotation definitions.`,
        false,
      );
      annotationLoaded = true;
      if (elements.fileInput) {
        elements.fileInput.disabled = false;
      }
      await discoverClassificationDatasets();
    } catch (error) {
      annotationSpecs = [];
      annotationById = {};
      annotationLoaded = false;
      const runningFromFile =
        typeof window !== "undefined" &&
        window.location &&
        window.location.protocol === "file:";
      const message = runningFromFile
        ? "Automatic annotation metadata loading is blocked for file:// pages. Start a local HTTP server (make viewer) so annotations.csv can be fetched."
        : "Unable to load annotations.csv automatically. Ensure the server is running and annotations.csv is reachable.";
      setAnnotationStatus(message, true);
      if (elements.fileInput) {
        elements.fileInput.disabled = true;
      }
      setLoaderStatus(
        "Viewer disabled: annotation metadata failed to load. Run `make viewer` from the repo root and reload once the server is available.",
        true,
      );
      console.error(error);
    }
  }

  /**
   * Discover available Parquet-backed classification datasets.
   */
  async function discoverClassificationDatasets() {
    const select = elements.datasetSelect;
    if (!select) {
      return;
    }
    select.disabled = true;
    select.innerHTML = '<option value="">Scanning annotations/…</option>';
    try {
      const response = await fetch("/api/classify-datasets", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      const items = Array.isArray(data.datasets) ? data.datasets : [];
      console.log("[viewer] classify-datasets response", {
        count: items.length,
        keys: items.map((item) => item && item.key).filter(Boolean),
      });
      if (!items.length) {
        setLoaderStatus(
          "No preprocessed Parquet datasets found under annotations/.",
          false,
        );
        select.innerHTML =
          '<option value="">No Parquet-backed datasets detected</option>';
        select.disabled = true;
        return;
      }
      const validItems = items
        .filter(
          (item) =>
            item && item.key && (item.preprocessed_path || item.matches_path),
        )
        .map((item) => ({
          key: String(item.key || "").trim(),
          label: applyDatasetLabelOrFallback(item, item.key),
          preprocessedPath: String(item.preprocessed_path || "").trim(),
          matchesPath: String(item.matches_path || "").trim(),
        }))
        .filter((item) => item.key);
      if (!validItems.length) {
        setLoaderStatus(
          "No usable Parquet-backed classification datasets found.",
          false,
        );
        select.innerHTML =
          '<option value="">No Parquet-backed datasets detected</option>';
        select.disabled = true;
        return;
      }
      validItems.sort((a, b) => collator.compare(b.key, a.key));
      state.datasets = validItems;
      select.innerHTML =
        '<option value="">Select a classification dataset</option>';
      state.datasets.forEach((ds) => {
        const option = document.createElement("option");
        option.value = ds.key;
        option.textContent = applyDatasetLabelOrFallback(ds, ds.key);
        select.appendChild(option);
      });
      select.disabled = false;
      let initialKey = "";
      try {
        if (window.localStorage) {
          initialKey =
            window.localStorage.getItem("classificationDatasetKey") || "";
        }
      } catch {
        initialKey = "";
      }
      if (!initialKey) {
        initialKey = state.datasets[0] ? state.datasets[0].key : "";
      }
      if (initialKey && state.datasets.some((ds) => ds.key === initialKey)) {
        select.value = initialKey;
        await loadDatasetMetadata(initialKey);
      }
    } catch (error) {
      console.error("Failed to load classification datasets:", error);
      select.innerHTML =
        '<option value="">Unable to scan annotations/ for Parquet datasets</option>';
      select.disabled = true;
    }
  }

  /**
   * Update the heading summary with file and record counts.
   */
  function updateSummary() {
    if (!elements.summary) {
      return;
    }
    if (!hasDataset()) {
      elements.summary.textContent =
        "Select a Parquet-backed dataset to get started.";
      return;
    }

    const dataset = state.dataset;
    const timestamp = new Date().toISOString();
    const total =
      typeof state.totalRecords === "number" && state.totalRecords >= 0
        ? state.totalRecords
        : 0;
    const sourceLabel = dataset.sourceId
      ? ` • Dataset: ${dataset.sourceId}`
      : "";
    elements.summary.innerHTML = `
      Loaded ${escapeHtml(String(total))} records<br>
      ${escapeHtml(sourceLabel)} • Queried at ${escapeHtml(timestamp)}
    `;
  }

  /**
   * Reset select menus back to their default option.
   */
  function clearSelectOptions(selectElement) {
    while (selectElement.options.length > 1) {
      selectElement.remove(1);
    }
  }

  /**
   * Populate the participant filter based on the active dataset.
   */
  function updateParticipantOptions() {
    const select = elements.participantSelect;
    if (!select) {
      return;
    }
    select.disabled = !hasDataset();
    clearSelectOptions(select);
    if (!hasDataset()) {
      select.value = "__all__";
      return;
    }
    const participants = Array.isArray(state.dataset.participants)
      ? state.dataset.participants
      : [];
    participants.forEach((participant) => {
      const option = document.createElement("option");
      option.value = participant;
      option.textContent = participant;
      select.appendChild(option);
    });
    if (
      state.participant !== "__all__" &&
      !state.dataset.participants.includes(state.participant)
    ) {
      state.participant = "__all__";
      select.value = "__all__";
    }
  }

  /**
   * Retrieve annotation metadata from either annotations.csv or records.
   */
  function getAnnotationSpec(annotationId) {
    if (!annotationId || annotationId === "__none__") {
      return null;
    }
    const combinedLookup = {
      ...(annotationById || {}),
      ...((state.dataset && state.dataset.annotationLookup) || {}),
    };
    return resolveAnnotationSpec(annotationId, annotationSpecs, combinedLookup);
  }

  /**
   * Populate annotation filter options from the active dataset.
   */
  function updateAnnotationOptions() {
    const select = elements.annotationSelect;
    if (!select) {
      return;
    }
    select.disabled = !hasDataset();
    // Rebuild options with category groupings and text "id: name"
    // Start fresh but preserve the first placeholder option
    while (select.options.length > 1) {
      select.remove(1);
    }
    if (!hasDataset()) {
      select.value = "__none__";
      return;
    }

    let annotationIds = Array.isArray(state.dataset.annotationIds)
      ? state.dataset.annotationIds.slice()
      : [];

    // Optionally filter annotations by scope (user / assistant).
    const scopeFilter = String(state.scopeFilter || "__all__").toLowerCase();
    if (scopeFilter !== "__all__" && Array.isArray(annotationSpecs)) {
      const allowedIds = new Set(
        annotationSpecs
          .filter(
            (spec) =>
              spec &&
              spec.id &&
              Array.isArray(spec.scope) &&
              spec.scope.includes(scopeFilter),
          )
          .map((spec) => spec.id),
      );
      annotationIds = annotationIds.filter((id) => allowedIds.has(id));
    }

    const dsLookup = state.dataset.annotationLookup || {};
    state.annotation = populateGroupedAnnotationOptions(
      select,
      annotationIds,
      annotationSpecs,
      dsLookup,
      state.annotation,
      "__none__",
    );
  }

  /**
   * Render the currently selected annotation details panel.
   */
  function updateAnnotationDetails() {
    const container = elements.annotationDetails;
    if (!container) {
      return;
    }
    const spec = getAnnotationSpec(state.annotation);
    if (!spec) {
      container.hidden = true;
      container.innerHTML = "";
      return;
    }
    renderAnnotationDetails(spec);
  }

  // Render read-only details for the selected annotation with snapshot info and actions.
  function renderAnnotationDetails(spec) {
    const container = elements.annotationDetails;
    const scope =
      spec.scope && spec.scope.length ? spec.scope.join(", ") : "Any role";
    const currentName = spec.name || "";
    const currentDesc = spec.description || "";

    const posExamples = Array.isArray(spec.positive_examples)
      ? spec.positive_examples
      : [];
    const negExamples = Array.isArray(spec.negative_examples)
      ? spec.negative_examples
      : [];

    const posBlock = posExamples.length
      ? `<details><summary>Positive examples (${posExamples.length})</summary><pre class="pre-box">${escapeHtml(
          posExamples.join("\n"),
        )}</pre></details>`
      : "";
    const negBlock = negExamples.length
      ? `<details><summary>Negative examples (${negExamples.length})</summary><pre class="pre-box">${escapeHtml(
          negExamples.join("\n"),
        )}</pre></details>`
      : "";

    container.hidden = false;
    container.innerHTML = `
      <h2>${escapeHtml(currentName || spec.id)}</h2>
      <p><strong>ID:</strong> ${escapeHtml(spec.id)}</p>
      <p><strong>Scope:</strong> ${escapeHtml(scope)}</p>
      <p><strong>Current name:</strong> ${escapeHtml(currentName || "(empty)")}</p>
      <p><strong>Current description:</strong><br>${escapeHtml(currentDesc || "(empty)").replace(/\n/g, "<br>")}</p>
      ${posBlock}
      ${negBlock}
      <div class="actions" style="margin-top:8px;">
        <button id="edit-annotation-button">Edit…</button>
      </div>
    `;

    const editBtn = container.querySelector("#edit-annotation-button");
    if (editBtn) {
      editBtn.addEventListener("click", () => renderAnnotationEditor(spec));
    }
  }

  // Render the inline editor for the selected annotation; includes Save and Regenerate.
  function renderAnnotationEditor(spec) {
    const container = elements.annotationDetails;
    const scope =
      spec.scope && spec.scope.length ? spec.scope.join(", ") : "Any role";
    const nameValue = spec.name || "";
    const descriptionValue = spec.description || "";
    const posValue = Array.isArray(spec.positive_examples)
      ? spec.positive_examples.join("\n")
      : "";
    const negValue = Array.isArray(spec.negative_examples)
      ? spec.negative_examples.join("\n")
      : "";

    const editor = `
      <h2>Edit Annotation</h2>
      <div class="form-grid">
        <label>ID</label>
        <div>${escapeHtml(spec.id)}</div>

        <label>Category</label>
        <div>${escapeHtml(spec.category || "Uncategorized")}</div>

        <label>Scope</label>
        <div>${escapeHtml(scope)}</div>

        <label for="annotation-name-input">Name</label>
        <input type="text" id="annotation-name-input" value="${escapeHtml(nameValue)}" />

        <label for="annotation-description-input">Description</label>
        <textarea id="annotation-description-input" rows="5">${escapeHtml(descriptionValue)}</textarea>

      <label for="annotation-positive-examples">Positive examples</label>
      <textarea id="annotation-positive-examples" rows="5" placeholder="One example per line">${escapeHtml(
        posValue,
      )}</textarea>
        <span class="control-note">Enter one example per line. Newlines separate examples; blank lines are ignored.</span>

      <label for="annotation-negative-examples">Negative examples</label>
      <textarea id="annotation-negative-examples" rows="5" placeholder="One example per line">${escapeHtml(
        negValue,
      )}</textarea>
        <span class="control-note">Enter one example per line. Newlines separate examples; blank lines are ignored.</span>
      </div>
      <div class="actions" style="margin-top:12px;">
        <button id="save-annotation-button">Save</button>
      </div>
    `;
    container.hidden = false;
    container.innerHTML = editor;

    const saveButton = container.querySelector("#save-annotation-button");
    if (saveButton) {
      saveButton.addEventListener("click", async () => {
        const nameInput = container.querySelector("#annotation-name-input");
        const descInput = container.querySelector(
          "#annotation-description-input",
        );
        const posInput = container.querySelector(
          "#annotation-positive-examples",
        );
        const negInput = container.querySelector(
          "#annotation-negative-examples",
        );
        const name = nameInput ? String(nameInput.value || "").trim() : "";
        const description = descInput
          ? String(descInput.value || "").trim()
          : "";
        const positive_examples = posInput
          ? normalizeLineBreaks(posInput.value)
          : "";
        const negative_examples = negInput
          ? normalizeLineBreaks(negInput.value)
          : "";
        await saveAnnotationEdits(
          spec.id,
          name,
          description,
          positive_examples,
          negative_examples,
        );
      });
    }

    const nameInput = container.querySelector("#annotation-name-input");
    const descInput = container.querySelector("#annotation-description-input");
    const posInput = container.querySelector("#annotation-positive-examples");
    const negInput = container.querySelector("#annotation-negative-examples");
    const computeDirty = () => {
      const currentName = nameInput ? String(nameInput.value || "").trim() : "";
      const currentDesc = descInput ? String(descInput.value || "").trim() : "";
      const currentPos = posInput ? normalizeLineBreaks(posInput.value) : "";
      const currentNeg = negInput ? normalizeLineBreaks(negInput.value) : "";
      const specPos = Array.isArray(spec.positive_examples)
        ? spec.positive_examples.join("\n").trim()
        : "";
      const specNeg = Array.isArray(spec.negative_examples)
        ? spec.negative_examples.join("\n").trim()
        : "";
      return (
        currentName !== (spec.name || "") ||
        currentDesc !== (spec.description || "") ||
        currentPos !== specPos ||
        currentNeg !== specNeg
      );
    };
    if (nameInput) nameInput.addEventListener("input", computeDirty);
    if (descInput) descInput.addEventListener("input", computeDirty);
    if (posInput) posInput.addEventListener("input", computeDirty);
    if (negInput) negInput.addEventListener("input", computeDirty);
    // Initialize state on first render
    computeDirty();
  }

  // Persist edits to annotations.csv via the local API and refresh metadata.
  async function saveAnnotationEdits(
    id,
    name,
    description,
    positive_examples,
    negative_examples,
  ) {
    try {
      const response = await fetch("/api/update-annotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id,
          name,
          description,
          positive_examples,
          negative_examples,
        }),
      });
      const data = await response.json();
      if (!response.ok || !data || data.error) {
        throw new Error(
          data && data.error ? data.error : `HTTP ${response.status}`,
        );
      }
      // Reload annotation metadata so UI stays in sync
      await loadAnnotationSpecs();
      // Keep the same selection
      updateAnnotationDetails();
    } catch (error) {
      console.error("Failed to save annotation edits:", error);
      alert(
        `Failed to save annotations: ${error && error.message ? error.message : error}`,
      );
    }
  }

  /**
   * Render the match list for a record as a small highlight block.
   */
  function renderMatches(record) {
    if (!record.has_match || !record.matches.length) {
      return "";
    }
    return renderMatchList(record.matches, "Matches:");
  }

  /**
   * Render the rationale block for a record when available.
   */
  function renderRationale(record) {
    const rationale =
      typeof record.rationale === "string" ? record.rationale.trim() : "";
    if (!rationale) {
      return "";
    }
    return renderTextBlock("Rationale", rationale);
  }

  function renderReasoning(record) {
    const content =
      typeof record.reasoning_content === "string"
        ? record.reasoning_content.trim()
        : "";
    if (!content || content === record.rationale) {
      return "";
    }
    return renderTextBlock("Reasoning content", content);
  }

  function renderRecord(record) {
    const cardClass = record.has_match ? "matching" : "non-matching";
    const statusLabel = record.has_match ? "Matching" : "Not matching";
    const metaParts = [];
    if (record.score_display != null) {
      metaParts.push(`Score: ${escapeHtml(String(record.score_display))}`);
    }
    if (record.model) {
      metaParts.push(`Model: ${escapeHtml(String(record.model))}`);
    }
    if (record.participant) {
      metaParts.push(`Participant: ${escapeHtml(record.participant)}`);
    }
    if (record.chat_key) {
      metaParts.push(`Chat: ${escapeHtml(record.chat_key)}`);
    }
    if (record.chat_date) {
      metaParts.push(`Date: ${escapeHtml(record.chat_date)}`);
    }
    if (record.role) {
      metaParts.push(`Role: ${escapeHtml(record.role)}`);
    }
    if (record.timestamp) {
      metaParts.push(`Message time: ${escapeHtml(record.timestamp)}`);
    }
    const metaLine = metaParts.join(" • ");
    const errorLine = record.error
      ? `<div class="alert">Error: ${escapeHtml(record.error)}</div>`
      : "";
    const contentHtml = renderMessageContent(
      record.content,
      state.showRawContent,
    );
    const sourceLabel = escapeHtml(
      record.source_path || record.source_file || "",
    );
    const sourceParts = [];
    if (sourceLabel) {
      sourceParts.push(sourceLabel);
    }
    const sourceDisplay = sourceParts.length
      ? sourceParts.join(" • ")
      : "Unknown";
    const hasContextReference =
      typeof record.chat_index === "number" &&
      typeof record.message_index === "number";
    let contextDataAttrs = "";
    if (hasContextReference) {
      const recordKey = `record-${recordCacheCounter}`;
      recordCacheCounter += 1;
      recordCache.set(recordKey, record);
      contextDataAttrs = ` data-record-id="${escapeHtml(recordKey)}"`;
    }
    const contextControls = renderContextControls(record);

    const rationaleHtml = renderRationale(record);
    const reasoningHtml = renderReasoning(record);

    return `<article class="result-card ${cardClass}">
      <div class="result-meta"><strong>${statusLabel}</strong>${
        metaLine ? ` • ${metaLine}` : ""
      }</div>
      <div class="message-with-context"${contextDataAttrs}>
        <div class="context-block context-block-before" data-context="before" hidden></div>
        <div class="message">${contentHtml}</div>
        <div class="context-block context-block-after" data-context="after" hidden></div>
      </div>
      ${rationaleHtml}
      ${reasoningHtml}
      ${renderMatches(record)}
      <div class="muted">Source: ${sourceDisplay}</div>
      ${contextControls}
      ${errorLine}
    </article>`;
  }

  async function getContextMessages(record, depth) {
    if (!record) {
      return { previous: [], next: [] };
    }

    const limit = clampContextDepth(depth);
    if (!limit) {
      return { previous: [], next: [] };
    }

    const sourcePath = record.source_path || record.source_file || "";
    const chatIndex =
      typeof record.chat_index === "number" ? record.chat_index : Number.NaN;
    const messageIndex =
      typeof record.message_index === "number"
        ? record.message_index
        : Number.NaN;

    if (!sourcePath || Number.isNaN(chatIndex) || Number.isNaN(messageIndex)) {
      throw new Error("Missing context reference fields on record");
    }

    const response = await fetch("/api/context-messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        participant: record.participant || "",
        source_path: sourcePath,
        chat_index: chatIndex,
        message_index: messageIndex,
        depth: limit,
      }),
      cache: "no-store",
    });

    let data;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok || !data || data.error) {
      const message =
        (data && data.error) || `HTTP ${response.status} while loading context`;
      throw new Error(message);
    }

    const previous = Array.isArray(data.previous) ? data.previous : [];
    const next = Array.isArray(data.next) ? data.next : [];
    return { previous, next };
  }

  function renderContextControls(record) {
    if (
      typeof record.chat_index !== "number" ||
      typeof record.message_index !== "number" ||
      (!record.source_path && !record.source_file)
    ) {
      return "";
    }
    return `<div class="context-controls">
      <button type="button" class="context-toggle">Show context</button>
    </div>`;
  }

  /**
   * Render the paginated result list or a relevant empty state.
   */
  function renderResults() {
    const container = elements.results;
    const alert = elements.resultsAlert;
    if (!container || !alert) {
      return;
    }

    if (!hasDataset()) {
      container.innerHTML = "";
      alert.textContent = "Select a Parquet-backed dataset to load messages.";
      alert.hidden = false;
      container.appendChild(alert);
      return;
    }

    if (state.annotation === "__none__") {
      container.innerHTML = "";
      alert.textContent = "Select an annotation to view messages.";
      alert.hidden = false;
      container.appendChild(alert);
      return;
    }

    alert.hidden = true;
    const rawRecords = Array.isArray(state.records) ? state.records : [];

    // Apply optional scope-based role filtering for the current view.
    let records = rawRecords;
    const scopeFilter = String(state.scopeFilter || "__all__").toLowerCase();
    if (scopeFilter === "user") {
      records = rawRecords.filter(
        (record) => String(record.role || "").toLowerCase() === "user",
      );
    } else if (scopeFilter === "assistant") {
      records = rawRecords.filter(
        (record) => String(record.role || "").toLowerCase() === "assistant",
      );
    }

    const keywordFilter = String(state.keywordFilter || "").trim();
    if (keywordFilter) {
      const normalizedKeyword = keywordFilter.toLowerCase();
      records = records.filter((record) => {
        const content = record && record.content ? String(record.content) : "";
        if (!content) {
          return false;
        }
        return content.toLowerCase().includes(normalizedKeyword);
      });
    }

    const totalRecords =
      keywordFilter && records.length
        ? records.length
        : typeof state.totalRecords === "number" && state.totalRecords >= 0
          ? state.totalRecords
          : records.length;
    if (!records.length || totalRecords === 0) {
      console.log("[viewer] renderResults: no records to display", {
        hasDataset: hasDataset(),
        annotation: state.annotation,
        participant: state.participant,
        total: totalRecords,
        page: state.page,
      });
      container.innerHTML =
        '<div class="alert">No records match the current filters.</div>';
      return;
    }

    const totalPages = Math.max(1, Math.ceil(totalRecords / state.pageSize));
    if (state.page > totalPages) {
      state.page = totalPages;
    }

    const startIndex = (state.page - 1) * state.pageSize;
    const start = Math.min(startIndex + 1, totalRecords);
    const end = Math.min(startIndex + state.pageSize, totalRecords);
    const visible = records;
    const pageInfo = `<span>Showing ${start}-${end} of ${totalRecords}</span>`;
    const pagination = `<div class="pagination">
      ${pageInfo}
      <div>
        <button id="prev-page" ${state.page === 1 ? "disabled" : ""}>Previous</button>
        <button id="next-page" ${
          state.page === totalPages ? "disabled" : ""
        }>Next</button>
      </div>
    </div>`;

    recordCache.clear();
    recordCacheCounter = 1;
    container.innerHTML =
      pagination + visible.map(renderRecord).join("") + pagination;

    const prevButtons = document.querySelectorAll("#prev-page");
    prevButtons.forEach((button) => {
      button.addEventListener("click", () => {
        if (state.page > 1) {
          state.page -= 1;
          loadCurrentPage();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      });
    });
    const nextButtons = document.querySelectorAll("#next-page");
    nextButtons.forEach((button) => {
      button.addEventListener("click", () => {
        if (state.page < totalPages) {
          state.page += 1;
          loadCurrentPage();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      });
    });
    attachContextHandlers(container);
  }

  function attachContextHandlers(container) {
    if (!container) {
      return;
    }
    const contextStateAttribute = "data-context-open";
    const labelShowContext = "Show context";
    const labelHideContext = "Hide context";
    const buttons = container.querySelectorAll(".context-toggle");
    buttons.forEach((button) => {
      button.addEventListener("click", async () => {
        const card = button.closest(".result-card");
        if (!card) {
          return;
        }
        const wrapper = card.querySelector(".message-with-context");
        if (!wrapper) {
          return;
        }
        const recordId = wrapper.getAttribute("data-record-id");
        if (!recordId || !recordCache.has(recordId)) {
          return;
        }
        const beforeBlock = wrapper.querySelector('[data-context="before"]');
        const afterBlock = wrapper.querySelector('[data-context="after"]');
        if (!beforeBlock || !afterBlock) {
          return;
        }
        const isOpen = wrapper.getAttribute(contextStateAttribute) === "true";
        if (isOpen) {
          clearContextBlock(beforeBlock);
          clearContextBlock(afterBlock);
          wrapper.setAttribute(contextStateAttribute, "false");
          button.textContent = labelShowContext;
          return;
        }
        const depth = clampContextDepth(state.contextDepth);
        if (!depth) {
          showContextInfo(
            beforeBlock,
            "Context disabled. Increase the window to see nearby messages.",
          );
          clearContextBlock(afterBlock);
          wrapper.setAttribute(contextStateAttribute, "true");
          button.textContent = labelHideContext;
          return;
        }
        button.disabled = true;
        const originalLabel = button.textContent;
        button.textContent = "Loading…";
        try {
          const context = await getContextMessages(
            recordCache.get(recordId),
            depth,
          );
          updateContextBlock(beforeBlock, "Before", context.previous, "before");
          updateContextBlock(afterBlock, "After", context.next, "after");
          wrapper.setAttribute(contextStateAttribute, "true");
          button.textContent = labelHideContext;
        } catch (error) {
          console.error("Unable to load context:", error);
          showContextError(beforeBlock, "Unable to load context messages.");
          clearContextBlock(afterBlock);
          wrapper.setAttribute(contextStateAttribute, "true");
          button.textContent = labelHideContext;
        } finally {
          button.disabled = false;
          if (wrapper.getAttribute(contextStateAttribute) !== "true") {
            button.textContent = originalLabel;
          }
        }
      });
    });
  }

  function clearContextBlock(element) {
    if (!element) {
      return;
    }
    element.innerHTML = "";
    element.hidden = true;
    element.classList.remove("is-empty");
  }

  function showContextError(element, message) {
    if (!element) {
      return;
    }
    element.innerHTML = `<div class="context-error">${escapeHtml(message)}</div>`;
    element.hidden = false;
    element.classList.add("is-empty");
  }

  function showContextInfo(element, message) {
    if (!element) {
      return;
    }
    element.innerHTML = `<div class="context-info">${escapeHtml(message)}</div>`;
    element.hidden = false;
    element.classList.add("is-empty");
  }

  function updateContextBlock(element, label, messages, variant) {
    if (!element) {
      return;
    }
    element.innerHTML = renderContextBlock(
      label,
      messages,
      variant,
      renderMessageContent,
    );
    element.hidden = false;
    if (Array.isArray(messages) && messages.length) {
      element.classList.remove("is-empty");
    } else {
      element.classList.add("is-empty");
    }
  }

  function setupContextDepthControl() {
    if (!elements.controls || elements.contextDepthSelect) {
      return;
    }
    const group = document.createElement("div");
    group.className = "control-group";
    group.innerHTML = `
      <label for="context-depth-select">Context window</label>
      <select id="context-depth-select"></select>
      <span class="control-note">Messages shown before and after when context is expanded.</span>
    `;
    const select = group.querySelector("#context-depth-select");
    if (select) {
      for (let i = 0; i <= 10; i += 1) {
        const option = document.createElement("option");
        option.value = String(i);
        option.textContent = i === 0 ? "0 (disabled)" : String(i);
        select.appendChild(option);
      }
      select.value = String(state.contextDepth);
    }
    elements.controls.appendChild(group);
    elements.contextDepthSelect = select;
  }

  function setupContentViewToggle() {
    if (!elements.controls || elements.showRawContentToggle) {
      return;
    }
    const group = document.createElement("div");
    group.className = "control-group";
    group.innerHTML = `
      <span style="font-weight: 600;">Content view</span>
      <div class="checkboxes">
        <label><input type="checkbox" id="show-raw-content"> Show original text</label>
      </div>
    `;
    elements.controls.appendChild(group);
    elements.showRawContentToggle = group.querySelector("#show-raw-content");
  }

  function setupErrorFilters() {
    // Error filters are not used for Parquet-backed datasets.
    return;
  }

  function setupScoreCutoffControl() {
    if (!elements.controls || elements.scoreCutoffInput) {
      return;
    }
    elements.scoreCutoffInput = installScoreCutoffControl(
      elements.controls,
      state.scoreCutoff,
      (clamped) => {
        state.scoreCutoff = clamped;
        state.page = 1;
        loadCurrentPage();
      },
    );
    const input = elements.scoreCutoffInput;
    if (input) {
      const group = input.closest(".control-group");
      if (group) {
        const modeSelect = document.createElement("select");
        modeSelect.id = "score-mode-select";
        modeSelect.innerHTML = `
          <option value="ge">\u2265 cutoff</option>
          <option value="eq">= cutoff only</option>
        `;
        modeSelect.value = state.scoreMode === "eq" ? "eq" : "ge";
        modeSelect.style.marginLeft = "8px";
        input.insertAdjacentElement("afterend", modeSelect);
        modeSelect.addEventListener("change", (event) => {
          const value =
            event && event.target && event.target.value === "eq" ? "eq" : "ge";
          state.scoreMode = value;
          state.page = 1;
          loadCurrentPage();
        });
      }
    }
  }

  /**
   * Normalize record shape for consistent downstream rendering.
   */
  function normalizeRecord(raw, sourceLabel) {
    const matches = Array.isArray(raw.matches) ? raw.matches : [];
    const annotationId = raw.annotation_id ? String(raw.annotation_id) : "";
    const rawRationale =
      typeof raw.rationale === "string" ? raw.rationale : null;
    const reasoningContent =
      typeof raw.reasoning_content === "string" ? raw.reasoning_content : null;
    const rawScore =
      typeof raw.score === "number" && Number.isFinite(raw.score)
        ? Math.round(raw.score)
        : null;
    let hasMatch = false;
    if (rawScore != null) {
      if (state.scoreMode === "eq") {
        hasMatch = rawScore === state.scoreCutoff;
      } else {
        hasMatch = rawScore >= state.scoreCutoff;
      }
    } else {
      hasMatch = matches.length > 0;
    }
    return {
      participant: raw.participant ? String(raw.participant) : "",
      annotation_id: annotationId,
      annotation_label: raw.annotation ? String(raw.annotation) : annotationId,
      chat_key: raw.chat_key,
      chat_date: raw.chat_date,
      chat_index: raw.chat_index,
      message_index: raw.message_index,
      role: raw.role,
      timestamp: raw.timestamp,
      matches,
      match_count: matches.length,
      has_match: hasMatch,
      score: rawScore,
      score_display: rawScore != null ? String(rawScore) : null,
      model: raw.model,
      error: raw.error,
      content: raw.content,
      source_path: raw.source_path,
      source_file: raw.source_file || sourceLabel,
      rationale: rawRationale,
      reasoning_content: reasoningContent,
    };
  }

  /**
   * Attach event listeners for the filters and controls.
   */
  function attachHandlers() {
    if (elements.datasetSelect) {
      elements.datasetSelect.addEventListener("change", async (event) => {
        const key = String(event.target.value || "").trim();
        state.datasetKey = key || null;
        state.page = 1;
        if (!key) {
          state.dataset = null;
          state.records = [];
          state.totalRecords = 0;
          updateSummary();
          renderResults();
          return;
        }
        await loadDatasetMetadata(key);
      });
    }
    if (elements.participantSelect) {
      elements.participantSelect.addEventListener("change", (event) => {
        state.participant = event.target.value;
        state.page = 1;
        updateAnnotationDetails();
        loadCurrentPage();
      });
    }
    if (elements.annotationSelect) {
      elements.annotationSelect.addEventListener("change", (event) => {
        state.annotation = event.target.value;
        state.page = 1;
        updateAnnotationDetails();
        loadCurrentPage();
      });
    }
    if (elements.pageSizeSelect) {
      elements.pageSizeSelect.addEventListener("change", (event) => {
        state.pageSize = Number(event.target.value);
        state.page = 1;
        loadCurrentPage();
      });
    }
    if (elements.scopeFilterSelect) {
      elements.scopeFilterSelect.addEventListener("change", (event) => {
        const value = String(event.target.value || "__all__").toLowerCase();
        state.scopeFilter =
          value === "user" || value === "assistant" ? value : "__all__";
        // Rebuild annotation options when scope filter changes.
        updateAnnotationOptions();
        updateAnnotationDetails();
        state.page = 1;
        loadCurrentPage();
      });
    }
    if (elements.keywordFilterInput) {
      elements.keywordFilterInput.addEventListener("input", (event) => {
        state.keywordFilter = String(event.target.value || "");
        state.page = 1;
        renderResults();
      });
    }
    if (elements.showMatching) {
      elements.showMatching.addEventListener("change", (event) => {
        state.showMatching = event.target.checked;
        state.page = 1;
        loadCurrentPage();
      });
    }
    if (elements.showNonMatching) {
      elements.showNonMatching.addEventListener("change", (event) => {
        state.showNonMatching = event.target.checked;
        state.page = 1;
        loadCurrentPage();
      });
    }
    if (elements.contextDepthSelect) {
      elements.contextDepthSelect.addEventListener("change", (event) => {
        state.contextDepth = clampContextDepth(event.target.value);
        renderResults();
      });
    }
    if (elements.showRawContentToggle) {
      elements.showRawContentToggle.addEventListener("change", (event) => {
        state.showRawContent = event.target.checked;
        renderResults();
      });
    }
    // Error toggles are no-ops for Parquet-backed datasets; errors are not surfaced.
  }

  async function loadDatasetMetadata(datasetKey) {
    if (!annotationLoaded) {
      return;
    }
    const descriptor =
      state.datasets.find((ds) => ds && ds.key === datasetKey) || null;
    if (!descriptor) {
      return;
    }
    state.datasetKey = datasetKey;
    setLoaderStatus("Loading metadata for selected dataset…", false);
    try {
      const response = await fetch(
        `/api/classify-metadata?dataset=${encodeURIComponent(datasetKey)}`,
        { cache: "no-store" },
      );
      const data = await response.json();
      if (!response.ok || data.error) {
        const message =
          (data && data.error) ||
          `HTTP ${response.status} while loading metadata`;
        throw new Error(message);
      }
      const participants = Array.isArray(data.participants)
        ? data.participants.slice()
        : [];
      const annotationIds = Array.isArray(data.annotation_ids)
        ? data.annotation_ids.slice()
        : [];
      state.dataset = {
        key: datasetKey,
        sourceId: datasetKey,
        participants,
        annotationIds,
        annotationLookup: {},
      };
      state.cutoffsByAnnotation =
        data &&
        data.cutoffs_by_annotation &&
        typeof data.cutoffs_by_annotation === "object"
          ? data.cutoffs_by_annotation
          : {};
      console.log("[viewer] classify-metadata loaded", {
        key: datasetKey,
        participants: participants.length,
        annotationIds: annotationIds.length,
        hasMatches: !!data.has_matches,
      });
      state.participant = "__all__";
      state.annotation = "__none__";
      state.page = 1;
      updateParticipantOptions();
      updateAnnotationOptions();
      updateAnnotationDetails();
      updateSummary();
      if (elements.participantSelect) {
        elements.participantSelect.value = "__all__";
      }
      if (elements.annotationSelect) {
        elements.annotationSelect.value = "__none__";
      }
      try {
        if (window.localStorage) {
          window.localStorage.setItem("classificationDatasetKey", datasetKey);
        }
      } catch {
        // ignore storage errors
      }
      // Automatically load the first page once metadata and filters are ready.
      await loadCurrentPage();
    } catch (error) {
      console.error("Failed to load dataset metadata:", error);
      setLoaderStatus("Unable to load dataset metadata.", true);
    }
  }

  async function loadCurrentPage() {
    if (
      !annotationLoaded ||
      !state.datasetKey ||
      state.annotation === "__none__"
    ) {
      state.records = [];
      state.totalRecords = 0;
      renderResults();
      return;
    }
    const params = new URLSearchParams();
    params.set("dataset", state.datasetKey);
    params.set("annotation_id", state.annotation);
    params.set("participant", state.participant || "__all__");
    params.set("page", String(state.page || 1));
    params.set("page_size", String(state.pageSize || 25));
    // When cutoffs are available, prefer discrete score_cutoff values.
    const cutoffs = state.cutoffsByAnnotation[state.annotation] || [];
    if (Array.isArray(cutoffs) && cutoffs.length) {
      // Snap current scoreCutoff to the closest available value.
      let best = cutoffs[0];
      let bestDiff = Math.abs(best - state.scoreCutoff);
      for (let index = 1; index < cutoffs.length; index += 1) {
        const value = cutoffs[index];
        const diff = Math.abs(value - state.scoreCutoff);
        if (diff < bestDiff) {
          best = value;
          bestDiff = diff;
        }
      }
      params.set("score_cutoff", String(best));
    }
    params.set("score_mode", state.scoreMode === "eq" ? "eq" : "ge");
    console.log("[viewer] classify-records request", params.toString());
    setLoaderStatus("Loading classification records for current page…", false);
    try {
      const response = await fetch(
        `/api/classify-records?${params.toString()}`,
        { cache: "no-store" },
      );
      const data = await response.json();
      if (!response.ok || data.error) {
        const message =
          (data && data.error) ||
          `HTTP ${response.status} while loading records`;
        throw new Error(message);
      }
      const rawRecords = Array.isArray(data.records) ? data.records : [];
      const normalized = rawRecords.map((raw) =>
        normalizeRecord(raw, state.datasetKey || ""),
      );
      state.records = normalized;
      const totalValue =
        typeof data.total === "number" && data.total >= 0
          ? data.total
          : normalized.length;
      state.totalRecords = totalValue;
      console.log("[viewer] classify-records response", {
        total: totalValue,
        page: data.page,
        pageSize: data.page_size,
        returned: rawRecords.length,
      });
      setLoaderStatus(
        `Loaded page ${data.page || state.page} of ${totalValue} record(s).`,
        false,
      );
      updateSummary();
      renderResults();
    } catch (error) {
      console.error("Failed to load classification records:", error);
      state.records = [];
      state.totalRecords = 0;
      setLoaderStatus("Unable to load classification records.", true);
      renderResults();
    }
  }

  /**
   * Kick off the viewer once the DOM is ready.
   */
  function initialize() {
    installViewerNav("classification");
    getConfig()
      .then((config) => {
        const cutoff =
          config && typeof config.llm_score_cutoff === "number"
            ? Math.round(config.llm_score_cutoff)
            : null;
        if (cutoff !== null && cutoff >= 0 && cutoff <= 10) {
          state.scoreCutoff = cutoff;
        }
      })
      .catch(() => {
        // Ignore config load failures; fall back to default state.
      })
      .finally(() => {
        setupContextDepthControl();
        setupContentViewToggle();
        setupErrorFilters();
        setupScoreCutoffControl();
        attachHandlers();
        updateSummary();
        renderResults();
        loadAnnotationSpecs();
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
