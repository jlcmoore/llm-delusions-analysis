import {
  AGREEMENT_ROOT,
  ANNOTATIONS_CSV_URL,
  applyDatasetLabelOrFallback,
  escapeHtml,
  fetchDirectoryEntries,
  installScoreCutoffControl,
  installViewerNav,
  parseAnnotationCsv,
  populateGroupedAnnotationOptions,
  resolveAnnotationSpec,
  renderAnnotationSummaryHtml,
  renderContextBlock,
  renderJsonOrMarkdown,
  renderMatchList,
  renderTextBlock,
} from "./viewer_shared.js";

(function () {
  "use strict";

  const elements = {
    summary: document.getElementById("agreement-summary"),
    datasetSelect: document.getElementById("dataset-select"),
    annotationSelect: document.getElementById("agreement-annotation-select"),
    participantSelect: document.getElementById("agreement-participant-select"),
    showAgree: document.getElementById("show-agree"),
    showDisagree: document.getElementById("show-disagree"),
    showContextToggle: document.getElementById("agreement-show-context"),
    showMatchesToggle: document.getElementById("agreement-show-matches"),
    showExplanationsToggle: document.getElementById(
      "agreement-show-explanations",
    ),
    annotationDetails: document.getElementById("agreement-annotation-details"),
    pageSizeSelect: document.getElementById("agreement-page-size"),
    metrics: document.getElementById("agreement-metrics"),
    results: document.getElementById("agreement-results"),
    resultsAlert: document.getElementById("agreement-results-alert"),
    controls: document.querySelector(".controls"),
    scoreCutoffInput: null,
  };

  const state = {
    datasets: [],
    currentDatasetKey: "",
    records: [],
    metrics: null,
    annotationSpecs: [],
    annotationById: {},
    currentAnnotation: "__all__",
    currentParticipant: "__all__",
    showAgree: false,
    showDisagree: true,
    showContext: true,
    showMatches: true,
    showExplanations: true,
    scoreCutoff: 5,
    availableScoreCutoffs: [],
    isLoading: false,
    pageSize: 50,
    page: 1,
    disagreementByKey: new Map(),
    disagreementRecords: [],
  };

  function computeKappaFromConfusion(entry) {
    const n11 = Number(entry.tp || 0);
    const n00 = Number(entry.tn || 0);
    const n10 = Number(entry.fn || 0);
    const n01 = Number(entry.fp || 0);
    const total = n11 + n00 + n10 + n01;
    if (!Number.isFinite(total) || total <= 0) {
      return null;
    }
    const observed = (n11 + n00) / total;
    const pAYes = (n11 + n10) / total;
    const pANo = (n00 + n01) / total;
    const pBYes = (n11 + n01) / total;
    const pBNo = (n00 + n10) / total;
    const expected = pAYes * pBYes + pANo * pBNo;
    const denom = 1 - expected;
    if (denom <= 0) {
      return null;
    }
    return (observed - expected) / denom;
  }

  function computeRatesFromCounts(tp, fp, tn, fn) {
    const total = tp + fp + tn + fn;
    if (!Number.isFinite(total) || total <= 0) {
      return {
        accuracy: null,
        fnr: null,
        fpr: null,
        precision: null,
        recall: null,
        f1: null,
      };
    }
    const accuracy = (tp + tn) / total;
    const fnr = tp + fn > 0 ? fn / (tp + fn) : null;
    const fpr = tn + fp > 0 ? fp / (tn + fp) : null;
    const precision = tp + fp > 0 ? tp / (tp + fp) : null;
    const recall = tp + fn > 0 ? tp / (tp + fn) : null;
    let f1 = null;
    if (precision !== null && recall !== null && precision + recall > 0) {
      f1 = (2 * precision * recall) / (precision + recall);
    }
    return {
      accuracy,
      fnr,
      fpr,
      precision,
      recall,
      f1,
    };
  }

  function buildCaseKeyPrimary(record) {
    const datasetPath = String(record.dataset_path || "");
    const annotationId = String(record.annotation_id || "");
    const sequenceIndex = Number(record.sequence_index);
    const participant = String(record.participant || "");
    if (
      !datasetPath ||
      !annotationId ||
      !Number.isFinite(sequenceIndex) ||
      !participant
    ) {
      return null;
    }
    return `${datasetPath}|${annotationId}|${sequenceIndex}|${participant}`;
  }

  function buildCaseKeyTranscript(record) {
    const transcriptKey =
      record && typeof record.transcript_key === "object"
        ? record.transcript_key
        : {};
    const participant = String(
      transcriptKey.participant || record.participant || "",
    );
    const sourcePath = String(transcriptKey.source_path || "");
    const chatIndex = Number(
      transcriptKey.chat_index !== undefined
        ? transcriptKey.chat_index
        : record.chat_index,
    );
    const messageIndex = Number(
      transcriptKey.message_index !== undefined
        ? transcriptKey.message_index
        : record.message_index,
    );
    if (
      !participant ||
      !sourcePath ||
      !Number.isFinite(chatIndex) ||
      !Number.isFinite(messageIndex)
    ) {
      return null;
    }
    return `${participant}|${sourcePath}|${chatIndex}|${messageIndex}`;
  }

  function buildCaseKeysForRecord(record) {
    const keys = [];
    const primary = buildCaseKeyPrimary(record);
    if (primary) {
      keys.push(primary);
    }
    const transcript = buildCaseKeyTranscript(record);
    if (transcript) {
      keys.push(transcript);
    }
    return keys;
  }

  function renderDisagreementExplanation(entry) {
    if (!entry || typeof entry !== "object") {
      return "";
    }
    const analysis = entry.analysis || {};
    const adjudicated = analysis.adjudicated_label || "unknown";
    const rationale = analysis.rationale || "";
    const leans = analysis.leans_toward || "";
    const causes = Array.isArray(analysis.disagreement_causes)
      ? analysis.disagreement_causes
      : [];
    const policyMisinterpretation =
      analysis.policy_misinterpretation === true ? "Yes" : "No";
    const shouldChange = analysis.should_change_prompt === true ? "Yes" : "No";
    const promptChange = analysis.prompt_change_suggestion || null;
    const uncertainty =
      typeof analysis.uncertainty === "number"
        ? analysis.uncertainty.toFixed(2)
        : "n/a";
    const evidence = Array.isArray(analysis.evidence_quotes)
      ? analysis.evidence_quotes.filter(
          (q) => typeof q === "string" && q.trim(),
        )
      : [];

    const causesHtml = causes.length
      ? `<ul class="explanation-list">${causes
          .map((cause) => `<li>${escapeHtml(String(cause))}</li>`)
          .join("")}</ul>`
      : '<p class="muted">No disagreement causes provided.</p>';

    const evidenceHtml = evidence.length
      ? `<ul class="explanation-list">${evidence
          .map((quote) => `<li>${escapeHtml(String(quote))}</li>`)
          .join("")}</ul>`
      : '<p class="muted">No evidence quotes provided.</p>';

    const promptChangeHtml =
      shouldChange === "Yes" && promptChange
        ? `<p>${escapeHtml(String(promptChange))}</p>`
        : '<p class="muted">No prompt change suggested.</p>';

    return `
      <div class="disagreement-explanation">
        <h4>Disagreement explanation (LLM)</h4>
        <div class="explanation-meta">
          Adjudicated label: <strong>${escapeHtml(String(adjudicated))}</strong>
          ${leans ? `• Leans toward: ${escapeHtml(String(leans))}` : ""}
          • Uncertainty: ${escapeHtml(String(uncertainty))}
        </div>
        ${rationale ? `<p>${escapeHtml(String(rationale))}</p>` : ""}
        <div class="explanation-meta">
          Policy misinterpretation: ${escapeHtml(policyMisinterpretation)}
          • Prompt change suggested: ${escapeHtml(shouldChange)}
        </div>
        <div>
          <strong>Likely causes</strong>
          ${causesHtml}
        </div>
        <div>
          <strong>Evidence quotes</strong>
          ${evidenceHtml}
        </div>
        <div>
          <strong>Prompt change suggestion</strong>
          ${promptChangeHtml}
        </div>
      </div>
    `;
  }

  function renderLlmDetails(record) {
    const detailsByAnnotator = record.annotator_details || {};
    if (!detailsByAnnotator || typeof detailsByAnnotator !== "object") {
      return "";
    }
    const metrics = state.metrics || {};
    const annotators = Array.isArray(metrics.annotators)
      ? metrics.annotators
      : [];
    const items = annotators
      .filter((info) => String(info.kind || "") === "llm")
      .map((info) => {
        const name = String(info.name || "");
        if (!name) {
          return "";
        }
        const details = detailsByAnnotator[name];
        if (!details || typeof details !== "object") {
          return "";
        }
        const score = typeof details.score === "number" ? details.score : null;
        const scoreLine =
          score !== null
            ? `<div class="match-item">Score: ${escapeHtml(
                String(score),
              )}</div>`
            : "";
        const rationaleText =
          typeof details.rationale === "string" ? details.rationale : "";
        const reasoningText =
          typeof details.reasoning_content === "string"
            ? details.reasoning_content
            : "";
        const rationaleHtml = renderTextBlock(
          `${name} rationale`,
          rationaleText,
        );
        const reasoningHtml =
          reasoningText && reasoningText !== rationaleText
            ? renderTextBlock(`${name} reasoning content`, reasoningText)
            : "";
        if (!scoreLine && !rationaleHtml && !reasoningHtml) {
          return "";
        }
        return `
          <div class="matches">
            <strong>${escapeHtml(name)} details:</strong>
            ${scoreLine}
          </div>
          ${rationaleHtml}
          ${reasoningHtml}
        `;
      })
      .filter(Boolean)
      .join("");
    return items || "";
  }

  function renderInterAnnotatorTable(pairs, humanIaa, title, useZebra) {
    const safePairs = Array.isArray(pairs) ? pairs : [];
    if (!safePairs.length) {
      return "";
    }
    const tableClass = useZebra ? " metrics-table-zebra" : "";
    let html = "";
    if (title) {
      html += `<h3>${escapeHtml(title)}</h3>`;
    }
    html +=
      '<p class="muted">' +
      escapeHtml(
        'Definitions: Items = dataset messages that contribute labels to this row (for Humans, messages with at least two human labels; for LLM pairs, messages labeled by both models). Pos agree = all annotators label "yes"; Neg agree = all annotators label "no"; Pos disagree = mixed-label messages with more "yes" than "no"; Neg disagree = mixed-label messages with more "no" than "yes"; ties contribute to Items and Ties but not to Pos/Neg disagree. \u03ba is Fleiss\u2019 kappa for the multi-rater human row and Cohen\u2019s kappa for LLM pair rows.',
      ) +
      "</p>";
    html +=
      `<table class="metrics-table${tableClass}">` +
      "<thead><tr>" +
      "<th>Annotators</th>" +
      "<th>Items</th>" +
      "<th>Pos agree</th>" +
      "<th>Neg agree</th>" +
      "<th>Pos disagree</th>" +
      "<th>Neg disagree</th>" +
      "<th>Ties</th>" +
      "<th>Agreement</th>" +
      "<th>\u03ba</th>" +
      "</tr></thead><tbody>";
    let rowIndex = 0;

    const metrics = state.metrics || {};
    const annotators = Array.isArray(metrics.annotators)
      ? metrics.annotators
      : [];
    const kindByName = Object.fromEntries(
      annotators.map((info) => [
        String(info.name || ""),
        String(info.kind || "unknown"),
      ]),
    );

    const llmPairs = safePairs.filter((pair) => {
      const rawA = String(pair.annotator_a || "");
      const rawB = String(pair.annotator_b || "");
      const kindA = kindByName[rawA] || "unknown";
      const kindB = kindByName[rawB] || "unknown";
      return kindA === "llm" && kindB === "llm";
    });

    function appendIaaRow(labelText, tp, fp, tn, fn) {
      const total = tp + fp + tn + fn;
      if (!Number.isFinite(total) || total <= 0) {
        return;
      }
      const posAgree = tp;
      const negAgree = tn;
      const posDisagree = fn;
      const negDisagree = fp;
      const ties = 0;
      const rates = computeRatesFromCounts(tp, fp, tn, fn);
      const agreement =
        rates.accuracy !== null
          ? `${(rates.accuracy * 100).toFixed(1)}%`
          : "n/a";
      const kappaValue = computeKappaFromConfusion({
        tp,
        fp,
        tn,
        fn,
      });
      const kappa =
        typeof kappaValue === "number" ? kappaValue.toFixed(3) : "n/a";
      const rowClass = rowIndex % 2 === 0 ? "row-even" : "row-odd";
      rowIndex += 1;
      html += `<tr class="${rowClass}">`;
      html += `<td>${escapeHtml(labelText)}</td>`;
      html += `<td>${escapeHtml(String(total))}</td>`;
      html += `<td>${escapeHtml(String(posAgree))}</td>`;
      html += `<td>${escapeHtml(String(negAgree))}</td>`;
      html += `<td>${escapeHtml(String(posDisagree))}</td>`;
      html += `<td>${escapeHtml(String(negDisagree))}</td>`;
      html += `<td>${escapeHtml(String(ties))}</td>`;
      html += `<td>${escapeHtml(agreement)}</td>`;
      html += `<td>${escapeHtml(kappa)}</td>`;
      html += "</tr>";
    }

    // Multi-rater human inter-annotator agreement over all human annotators.
    if (humanIaa && typeof humanIaa === "object") {
      const nItems = Number(humanIaa.n_items || 0);
      if (Number.isFinite(nItems) && nItems > 0) {
        const posAgree = Number(humanIaa.pos_agree || 0);
        const negAgree = Number(humanIaa.neg_agree || 0);
        const posDisagree = Number(humanIaa.pos_disagree || 0);
        const negDisagree = Number(humanIaa.neg_disagree || 0);
        const totalDisagree = Number(humanIaa.disagree || 0);
        const ties = Math.max(0, totalDisagree - posDisagree - negDisagree);
        const agreement =
          typeof humanIaa.agreement === "number"
            ? `${(humanIaa.agreement * 100).toFixed(1)}%`
            : "n/a";
        const kappa =
          typeof humanIaa.kappa === "number"
            ? humanIaa.kappa.toFixed(3)
            : "n/a";
        const rowClass = rowIndex % 2 === 0 ? "row-even" : "row-odd";
        rowIndex += 1;
        html += `<tr class="${rowClass}">`;
        html += `<td>Humans (all annotators)</td>`;
        html += `<td>${escapeHtml(String(nItems))}</td>`;
        html += `<td>${escapeHtml(String(posAgree))}</td>`;
        html += `<td>${escapeHtml(String(negAgree))}</td>`;
        html += `<td>${escapeHtml(String(posDisagree))}</td>`;
        html += `<td>${escapeHtml(String(negDisagree))}</td>`;
        html += `<td>${escapeHtml(String(ties))}</td>`;
        html += `<td>${escapeHtml(agreement)}</td>`;
        html += `<td>${escapeHtml(kappa)}</td>`;
        html += "</tr>";
      }
    }

    // Show all pairwise agreements for LLMs.
    llmPairs.forEach((pair) => {
      const counts = pair.counts || {};
      const tp = Number(counts.yes_yes || 0);
      const tn = Number(counts.no_no || 0);
      const fn = Number(counts.yes_no || 0);
      const fp = Number(counts.no_yes || 0);
      const rawA = String(pair.annotator_a || "");
      const rawB = String(pair.annotator_b || "");
      const ordered = [rawA, rawB].sort((a, b) => a.localeCompare(b, "en"));
      const leftName = ordered[0];
      const rightName = ordered[1];
      const label = `${leftName} vs ${rightName}`;
      appendIaaRow(label, tp, fp, tn, fn);
    });

    html += "</tbody></table>";
    return html;
  }

  function renderMajorityTable(confusionEntries, title, useZebra) {
    const safeConfusion = Array.isArray(confusionEntries)
      ? confusionEntries
      : [];
    if (!safeConfusion.length) {
      return "";
    }
    const tableClass = useZebra ? " metrics-table-zebra" : "";
    let html = "";
    if (title) {
      html += `<h3>${escapeHtml(title)}</h3>`;
    }
    html +=
      '<p class="muted">' +
      escapeHtml(
        "Definitions: Items = dataset messages that have a non-tied human majority label and a label from the annotator in this row. TP, FP, TN, and FN are counted versus that human majority label.",
      ) +
      "</p>";
    html +=
      `<table class="metrics-table${tableClass}">` +
      "<thead><tr>" +
      "<th>Annotators</th>" +
      "<th>Items</th>" +
      "<th>TP</th>" +
      "<th>FP</th>" +
      "<th>TN</th>" +
      "<th>FN</th>" +
      "<th>FNR</th>" +
      "<th>FPR</th>" +
      "<th>Accuracy</th>" +
      "<th>Precision</th>" +
      "<th>Recall</th>" +
      "<th>F1</th>" +
      "<th>Cohen\u2019s \u03ba</th>" +
      "</tr></thead><tbody>";
    let rowIndex = 0;

    safeConfusion.forEach((entry) => {
      const tp = Number(entry.tp || 0);
      const fp = Number(entry.fp || 0);
      const tn = Number(entry.tn || 0);
      const fn = Number(entry.fn || 0);
      const total = tp + fp + tn + fn;
      if (!Number.isFinite(total) || total <= 0) {
        return;
      }
      const rates = computeRatesFromCounts(tp, fp, tn, fn);
      const fnr =
        rates.fnr !== null ? `${(rates.fnr * 100).toFixed(1)}%` : "n/a";
      const fpr =
        rates.fpr !== null ? `${(rates.fpr * 100).toFixed(1)}%` : "n/a";
      const accuracy =
        rates.accuracy !== null
          ? `${(rates.accuracy * 100).toFixed(1)}%`
          : "n/a";
      const precision =
        rates.precision !== null ? rates.precision.toFixed(3) : "n/a";
      const recall = rates.recall !== null ? rates.recall.toFixed(3) : "n/a";
      const f1 = rates.f1 !== null ? rates.f1.toFixed(3) : "n/a";
      const kappaValue = computeKappaFromConfusion({
        tp,
        fp,
        tn,
        fn,
      });
      const kappa =
        typeof kappaValue === "number" ? kappaValue.toFixed(3) : "n/a";
      const baseAnnotator = String(entry.annotator || "");
      const labelText = `${baseAnnotator} vs humans (majority)`;
      const labelHtml = `${escapeHtml(
        labelText,
      )} <span class="metrics-majority-badge">majority ref</span>`;
      const rowClass = rowIndex % 2 === 0 ? "row-even" : "row-odd";
      rowIndex += 1;
      html += `<tr class="${rowClass}">`;
      html += `<td class="metrics-majority-label">${labelHtml}</td>`;
      html += `<td>${escapeHtml(String(total))}</td>`;
      html += `<td class="cm-tp">${escapeHtml(String(tp))}</td>`;
      html += `<td class="cm-fp">${escapeHtml(String(fp))}</td>`;
      html += `<td class="cm-tn">${escapeHtml(String(tn))}</td>`;
      html += `<td class="cm-fn">${escapeHtml(String(fn))}</td>`;
      html += `<td>${escapeHtml(fnr)}</td>`;
      html += `<td>${escapeHtml(fpr)}</td>`;
      html += `<td>${escapeHtml(accuracy)}</td>`;
      html += `<td>${escapeHtml(precision)}</td>`;
      html += `<td>${escapeHtml(recall)}</td>`;
      html += `<td>${escapeHtml(f1)}</td>`;
      html += `<td>${escapeHtml(kappa)}</td>`;
      html += "</tr>";
    });

    html += "</tbody></table>";
    return html;
  }

  function setSummary(message) {
    if (!elements.summary) return;
    elements.summary.textContent = message;
  }

  function setResultsAlert(message, isError) {
    if (!elements.resultsAlert) return;
    elements.resultsAlert.textContent = message;
    elements.resultsAlert.classList.toggle("error", !!isError);
  }

  function setLoading(isLoading) {
    state.isLoading = !!isLoading;
    if (elements.datasetSelect) {
      elements.datasetSelect.disabled = isLoading;
    }
    if (elements.scoreCutoffInput) {
      elements.scoreCutoffInput.disabled = isLoading;
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
    } catch (error) {
      console.error(
        "Failed to load annotations.csv for agreement viewer:",
        error,
      );
      state.annotationSpecs = [];
      state.annotationById = {};
    }
  }

  function getAnnotationSpec(annotationId) {
    if (!annotationId || annotationId === "__all__") {
      return null;
    }
    return resolveAnnotationSpec(
      annotationId,
      state.annotationSpecs,
      state.annotationById,
    );
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
    const metrics = state.metrics || {};
    const cutoffsByAnnotation =
      metrics && typeof metrics === "object"
        ? metrics.llm_score_cutoffs_by_annotation || {}
        : {};
    let cutoffLine = "";
    if (
      state.currentAnnotation &&
      Object.prototype.hasOwnProperty.call(
        cutoffsByAnnotation,
        state.currentAnnotation,
      )
    ) {
      cutoffLine = `LLM score cutoff for this annotation: ${escapeHtml(
        String(cutoffsByAnnotation[state.currentAnnotation]),
      )}`;
    } else if (
      typeof metrics.llm_score_cutoff === "number" &&
      Number.isFinite(metrics.llm_score_cutoff)
    ) {
      cutoffLine = `LLM score cutoff (global): ${escapeHtml(
        String(metrics.llm_score_cutoff),
      )}`;
    }
    if (cutoffLine) {
      container.innerHTML += `<p><strong>${cutoffLine}</strong></p>`;
    }
  }

  async function discoverAgreementDatasets() {
    try {
      const response = await fetch("/api/agreement-datasets", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      const items = Array.isArray(data.datasets) ? data.datasets : [];
      if (!items.length) {
        setResultsAlert(
          "No agreement datasets found under analysis/agreement/. Run compute_annotation_agreement.py and reload.",
          false,
        );
        elements.datasetSelect.disabled = true;
        return;
      }
      state.datasets = items
        .filter((item) => item && item.key)
        .map((item) => {
          const key = String(item.key || "").trim();
          return { key, label: applyDatasetLabelOrFallback(item, key) };
        })
        .filter((ds) => ds.key);
      populateDatasetOptions();
    } catch (error) {
      console.error("Failed to scan analysis/ for agreement cases:", error);
      setResultsAlert(
        "Unable to scan analysis/ for agreement_cases files. Ensure directory listing is enabled.",
        true,
      );
      elements.datasetSelect.disabled = true;
    }
  }

  function populateDatasetOptions() {
    const select = elements.datasetSelect;
    if (!select) return;
    while (select.options.length) {
      select.remove(0);
    }
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select an agreement dataset";
    select.appendChild(placeholder);

    state.datasets.forEach((ds) => {
      const option = document.createElement("option");
      option.value = ds.key;
      option.textContent = applyDatasetLabelOrFallback(ds, ds.key);
      select.appendChild(option);
    });
    select.disabled = false;
  }

  async function loadCurrentDataset() {
    const ds = state.datasets.find(
      (entry) => entry.key === state.currentDatasetKey,
    );
    if (!ds) {
      state.records = [];
      state.metrics = null;
      renderMetrics();
      renderResults();
      return;
    }

    try {
      setLoading(true);
      setResultsAlert("Loading agreement dataset…", false);

      // Discover available score-specific metrics files for this dataset.
      const baseUrl = new URL(
        `${AGREEMENT_ROOT}${ds.key}/`,
        window.location.origin,
      );
      const dirEntries = await fetchDirectoryEntries(baseUrl.href);
      const cutoffs = Array.from(
        new Set(
          dirEntries
            .filter(
              (href) =>
                href.startsWith("metrics.score-") && href.endsWith(".json"),
            )
            .map((href) => {
              const match = href.match(/metrics\.score-(\d+)\.json$/);
              return match ? Number(match[1]) : null;
            })
            .filter(
              (value) => Number.isFinite(value) && value >= 0 && value <= 10,
            ),
        ),
      ).sort((a, b) => a - b);
      state.availableScoreCutoffs = cutoffs;
      if (!cutoffs.length) {
        throw new Error(
          `No metrics.score-*.json files found under ${AGREEMENT_ROOT}${ds.key}/`,
        );
      }

      let cutoff = Number.isFinite(Number(state.scoreCutoff))
        ? Number(state.scoreCutoff)
        : 5;
      if (!cutoffs.includes(cutoff)) {
        if (cutoffs.includes(5)) {
          cutoff = 5;
        } else {
          cutoff = cutoffs[0];
        }
        state.scoreCutoff = cutoff;
        if (elements.scoreCutoffInput) {
          elements.scoreCutoffInput.value = String(cutoff);
        }
      }

      const scoreSuffix = `.score-${cutoff}`;
      const casesPath = `${AGREEMENT_ROOT}${ds.key}/cases${scoreSuffix}.jsonl`;
      const metricsPath = `${AGREEMENT_ROOT}${ds.key}/metrics${scoreSuffix}.json`;
      const disagreementsPath = `${AGREEMENT_ROOT}${ds.key}/disagreements${scoreSuffix}.jsonl`;

      const [casesText, metricsResp, disagreementsText] = await Promise.all([
        fetch(casesPath, { cache: "no-store" }).then((r) => {
          if (!r.ok) {
            throw new Error(`HTTP ${r.status} for ${casesPath}`);
          }
          return r.text();
        }),
        fetch(metricsPath, { cache: "no-store" }).then((r) => {
          if (!r.ok) {
            throw new Error(`HTTP ${r.status} for ${metricsPath}`);
          }
          return r.json();
        }),
        fetch(disagreementsPath, { cache: "no-store" })
          .then((r) => {
            if (!r.ok) {
              return "";
            }
            return r.text();
          })
          .catch(() => ""),
      ]);

      const records = [];
      casesText
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .forEach((line) => {
          try {
            const obj = JSON.parse(line);
            if (obj && typeof obj === "object") {
              records.push(obj);
            }
          } catch (error) {
            console.error("Failed to parse agreement case line:", error);
          }
        });

      state.records = records;
      state.metrics = metricsResp || null;
      state.disagreementByKey = new Map();
      state.disagreementRecords = [];

      const disagreementRecords = [];
      disagreementsText
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .forEach((line) => {
          try {
            const obj = JSON.parse(line);
            if (obj && typeof obj === "object") {
              disagreementRecords.push(obj);
            }
          } catch (error) {
            console.error("Failed to parse disagreement analysis line:", error);
          }
        });

      const disagreementMap = new Map();
      disagreementRecords.forEach((record) => {
        const caseKeys = Array.isArray(record.case_keys)
          ? record.case_keys.filter(
              (value) => typeof value === "string" && value.trim(),
            )
          : buildCaseKeysForRecord(record);
        caseKeys.forEach((key) => {
          if (!disagreementMap.has(key)) {
            disagreementMap.set(key, record);
          }
        });
      });
      state.disagreementByKey = disagreementMap;
      state.disagreementRecords = disagreementRecords;

      updateAnnotationOptionsFromRecords();
      updateParticipantOptionsFromRecords();
      updateAnnotationDetails();
      renderMetrics();
      renderResults();

      const total = state.records.length;
      if (total) {
        const annotatorCount =
          (state.metrics && Array.isArray(state.metrics.annotators)
            ? state.metrics.annotators.length
            : 0) || 0;
        setSummary(
          `Loaded ${total} labeled messages across ${annotatorCount} annotator(s).`,
        );
      } else {
        setSummary("Loaded agreement dataset with no records.");
      }
      setLoading(false);
      setResultsAlert("", false);
    } catch (error) {
      console.error("Failed to load agreement dataset:", error);
      state.records = [];
      state.metrics = null;
      renderMetrics();
      renderResults();
      setLoading(false);
      setResultsAlert(
        "Failed to load agreement dataset. Check console for details.",
        true,
      );
    }
  }

  function updateAnnotationOptionsFromRecords() {
    const select = elements.annotationSelect;
    if (!select) {
      return;
    }

    const ids = Array.from(
      new Set(state.records.map((rec) => rec.annotation_id).filter(Boolean)),
    ).sort();

    // Preserve the first placeholder ("All annotations") and append grouped options.
    state.currentAnnotation = populateGroupedAnnotationOptions(
      select,
      ids,
      state.annotationSpecs,
      {},
      state.currentAnnotation,
      "__all__",
    );
    select.disabled = false;

    // When a new dataset is loaded, default to "All annotations" rather than
    // the lexicographically first annotation id.
    state.currentAnnotation = "__all__";
    select.value = "__all__";
    updateAnnotationDetails();
  }

  function updateParticipantOptionsFromRecords() {
    const select = elements.participantSelect;
    if (!select) return;
    while (select.options.length) {
      select.remove(0);
    }
    const allOption = document.createElement("option");
    allOption.value = "__all__";
    allOption.textContent = "All participants";
    select.appendChild(allOption);

    const participants = Array.from(
      new Set(state.records.map((rec) => rec.participant).filter(Boolean)),
    ).sort();
    participants.forEach((ppt) => {
      const option = document.createElement("option");
      option.value = ppt;
      option.textContent = ppt;
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

  function renderMetrics() {
    const container = elements.metrics;
    if (!container) return;
    const metrics = state.metrics;
    if (!metrics) {
      container.hidden = true;
      container.innerHTML = "";
      return;
    }
    const annotators = Array.isArray(metrics.annotators)
      ? metrics.annotators
      : [];
    const pairwiseByAnnotation = metrics.pairwise || {};
    const majorityByAnnotation = metrics.majority_confusion || {};
    const overallKey = "__all__";
    const targetAnnotation =
      state.currentAnnotation && state.currentAnnotation !== "__all__"
        ? state.currentAnnotation
        : null;
    const hasOverall = Boolean(pairwiseByAnnotation[overallKey]);
    const annotationIds = Object.keys(pairwiseByAnnotation)
      .filter((id) => id !== overallKey)
      .sort();

    let html = "";
    html += "<h2>Annotator agreement</h2>";
    html += "<p>";
    html += escapeHtml(
      "Pairwise percent agreement and Cohen's kappa for overlapping labels.",
    );
    html += "</p>";
    const cutoff =
      typeof metrics.llm_score_cutoff === "number" &&
      Number.isFinite(metrics.llm_score_cutoff)
        ? metrics.llm_score_cutoff
        : null;
    const cutoffsByAnnotation =
      metrics.llm_score_cutoffs_by_annotation &&
      typeof metrics.llm_score_cutoffs_by_annotation === "object"
        ? metrics.llm_score_cutoffs_by_annotation
        : {};
    const cutoffValues = Object.values(cutoffsByAnnotation)
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v));
    const uniqueCutoffs = Array.from(new Set(cutoffValues)).sort(
      (a, b) => a - b,
    );
    if (uniqueCutoffs.length > 1) {
      html += '<p class="muted">';
      html += escapeHtml(
        "LLM labels use per-annotation score cutoffs when computing " +
          "majority-based metrics. Each annotation row uses its own " +
          "F1-maximizing cutoff versus the human-majority labels.",
      );
      html += "</p>";
    } else if (cutoff !== null) {
      html += '<p class="muted">';
      html += escapeHtml(
        `LLM labels are binarized using a score cutoff of ${cutoff} (scores \u2265 ${cutoff} are treated as positive).`,
      );
      html += "</p>";
    }

    if (annotators.length) {
      const rows = annotators
        .map((info) => {
          const parts = [];
          parts.push(
            `<strong>${escapeHtml(String(info.name || ""))}</strong> (${escapeHtml(
              String(info.kind || "unknown"),
            )})`,
          );
          if (info.source) {
            parts.push(
              `<span class="wrap-path">${escapeHtml(String(info.source))}</span>`,
            );
          }
          return `<li>${parts.join("<br>")}</li>`;
        })
        .join("");
      html += `<details open><summary>Annotators</summary><ul>${rows}</ul></details>`;
    }

    const datasetLabel =
      typeof metrics.dataset === "string" && metrics.dataset
        ? metrics.dataset
        : "(unknown dataset)";
    html += `<p class="muted">Statistics are computed over all labeled items in <span class="wrap-path">${escapeHtml(
      datasetLabel,
    )}</span> for the active annotation filter and all participants present in that dataset. When showing all annotations, the Overall table pools items across all annotations in this dataset.</p>`;

    const humanIaaByAnnotation = metrics.human_iaa || {};

    if (!targetAnnotation && hasOverall) {
      const overallPairs = pairwiseByAnnotation[overallKey] || [];
      const overallConfusion = majorityByAnnotation[overallKey] || [];
      const overallHumanIaa = humanIaaByAnnotation[overallKey] || null;
      html += renderInterAnnotatorTable(
        overallPairs,
        overallHumanIaa,
        "Overall inter-annotator agreement (all annotations)",
        true,
      );
      html += renderMajorityTable(
        overallConfusion,
        "LLM vs human majority (all annotations)",
        true,
      );
    }

    const relevantIds = targetAnnotation ? [targetAnnotation] : annotationIds;
    relevantIds.forEach((annotationId) => {
      const pairs = pairwiseByAnnotation[annotationId] || [];
      const confusionEntries = majorityByAnnotation[annotationId] || [];
      if (!pairs.length && !confusionEntries.length) {
        return;
      }
      const spec = getAnnotationSpec(annotationId);
      const title =
        (spec && spec.name) || metrics.annotation_ids?.[0] || annotationId;
      const zebraClass = !targetAnnotation;
      const humanIaaEntry = humanIaaByAnnotation[annotationId] || null;
      const perAnnotationCutoff = Object.prototype.hasOwnProperty.call(
        cutoffsByAnnotation,
        annotationId,
      )
        ? Number(cutoffsByAnnotation[annotationId])
        : null;
      if (
        perAnnotationCutoff !== null &&
        Number.isFinite(perAnnotationCutoff)
      ) {
        const cutoffText = escapeHtml(String(perAnnotationCutoff));
        html +=
          '<p class="muted">For ' +
          escapeHtml(title) +
          " (" +
          escapeHtml(annotationId) +
          "), LLM labels use a per-annotation score cutoff of " +
          cutoffText +
          " (scores \u2265 " +
          cutoffText +
          " are treated as positive).</p>";
      }
      html += renderInterAnnotatorTable(
        pairs,
        humanIaaEntry,
        `Inter-annotator agreement for ${title} (${annotationId})`,
        zebraClass,
      );
      html += renderMajorityTable(
        confusionEntries,
        `LLM vs human majority for ${title} (${annotationId})`,
        zebraClass,
      );
    });

    container.hidden = false;
    container.innerHTML = html;
  }

  function classifyCase(record) {
    const labels = record.annotator_labels || {};
    const values = Object.values(labels).filter(
      (value) => value === "yes" || value === "no",
    );
    if (values.length < 2) {
      return "single";
    }
    const unique = Array.from(new Set(values));
    if (unique.length === 1) {
      return "agree";
    }
    return "disagree";
  }

  function filterRecords() {
    return state.records.filter((record) => {
      if (
        state.currentAnnotation !== "__all__" &&
        record.annotation_id !== state.currentAnnotation
      ) {
        return false;
      }
      if (
        state.currentParticipant !== "__all__" &&
        record.participant !== state.currentParticipant
      ) {
        return false;
      }
      const kind = classifyCase(record);
      if (kind === "agree" && !state.showAgree) {
        return false;
      }
      if (kind === "disagree" && !state.showDisagree) {
        return false;
      }
      if (kind === "single" && !state.showAgree && !state.showDisagree) {
        return false;
      }
      return true;
    });
  }

  function renderResults() {
    const container = elements.results;
    if (!container) return;

    const records = filterRecords();
    if (!records.length) {
      container.innerHTML =
        '<div class="alert" id="agreement-results-alert">No records match the current filters.</div>';
      return;
    }

    const totalPages = Math.max(1, Math.ceil(records.length / state.pageSize));
    if (state.page > totalPages) {
      state.page = totalPages;
    }

    const start = (state.page - 1) * state.pageSize;
    const visible = records.slice(start, start + state.pageSize);
    const pageInfo = `<span>Showing ${start + 1}-${Math.min(
      start + state.pageSize,
      records.length,
    )} of ${records.length}</span>`;
    const pagination = `<div class="pagination">
      ${pageInfo}
      <div>
        <button id="agreement-prev-page" ${
          state.page === 1 ? "disabled" : ""
        }>Previous</button>
        <button id="agreement-next-page" ${
          state.page === totalPages ? "disabled" : ""
        }>Next</button>
      </div>
    </div>`;

    const metrics = state.metrics || {};
    const annotators = Array.isArray(metrics.annotators)
      ? metrics.annotators
      : [];
    const annotatorNames = annotators.map((info) => String(info.name || ""));

    const cards = visible
      .map((record) => {
        const kind = classifyCase(record);
        const kindLabel =
          kind === "agree"
            ? "All annotators agree"
            : kind === "disagree"
              ? "Annotators disagree"
              : "Single annotator";
        const metaParts = [];
        if (record.participant) {
          metaParts.push(`Participant: ${escapeHtml(record.participant)}`);
        }
        if (record.chat_key) {
          metaParts.push(
            `Chat: <span class="wrap-path">${escapeHtml(record.chat_key)}</span>`,
          );
        }
        if (
          typeof record.message_index === "number" &&
          Number.isFinite(record.message_index)
        ) {
          metaParts.push(
            `Message index: ${escapeHtml(String(record.message_index))}`,
          );
        }

        const metaHtml = metaParts.length
          ? `<div class="result-meta">${metaParts.join(" \u2022 ")}</div>`
          : "";

        const annotationTitle = escapeHtml(
          record.annotation_label || record.annotation_id || "",
        );

        const preceding = Array.isArray(record.preceding)
          ? record.preceding
          : [];
        const ctxHtml = state.showContext
          ? renderContextBlock(
              "Preceding context",
              preceding,
              "before",
              renderJsonOrMarkdown,
            )
          : "";

        const annotatorRows = annotatorNames
          .map((name) => {
            if (!name) return "";
            const value = (record.annotator_labels || {})[name] || null;
            const label =
              value === "yes" ? "Yes" : value === "no" ? "No" : "Missing";
            const css =
              value === "yes"
                ? "label-yes"
                : value === "no"
                  ? "label-no"
                  : "label-missing";
            return `<tr><td>${escapeHtml(
              name,
            )}</td><td><span class="label-chip ${css}">${escapeHtml(
              label,
            )}</span></td></tr>`;
          })
          .filter(Boolean)
          .join("");

        let matchesHtml = "";
        if (state.showMatches) {
          const matchesByAnnotator = record.annotator_matches || {};
          const metricsForMatches = state.metrics || {};
          const annotatorInfos = Array.isArray(metricsForMatches.annotators)
            ? metricsForMatches.annotators
            : [];
          const kindLookup = Object.fromEntries(
            annotatorInfos.map((info) => [
              String(info.name || ""),
              String(info.kind || "unknown"),
            ]),
          );
          const items = Object.entries(matchesByAnnotator)
            .filter(([name, values]) => {
              const kind = kindLookup[String(name)] || "unknown";
              return (
                kind === "llm" &&
                Array.isArray(values) &&
                values.some((v) => typeof v === "string" && v.trim())
              );
            })
            .map(([name, values]) =>
              renderMatchList(values, `${String(name)} matches:`),
            )
            .filter(Boolean)
            .join("");
          matchesHtml = items || "";
        }

        const llmDetailsHtml = renderLlmDetails(record);
        const annotatorTable = annotatorRows
          ? `<table class="annotator-table"><thead><tr><th>Annotator</th><th>Label</th></tr></thead><tbody>${annotatorRows}</tbody></table>`
          : "<p>No annotator labels available.</p>";

        const messageHtml = renderJsonOrMarkdown(record.content || "");

        let explanationHtml = "";
        if (kind === "disagree" && state.showExplanations) {
          const keys = buildCaseKeysForRecord(record);
          let analysis = null;
          for (const key of keys) {
            if (state.disagreementByKey.has(key)) {
              analysis = state.disagreementByKey.get(key);
              break;
            }
          }
          if (analysis) {
            explanationHtml = renderDisagreementExplanation(analysis);
          }
        }

        return `
          <article class="result-card">
            <div class="result-meta">
              <strong>${escapeHtml(kindLabel)}</strong>
            </div>
            ${metaHtml}
            <h3>${annotationTitle}</h3>
            <div class="message-with-context">
              ${
                state.showContext
                  ? `<div class="context-block context-block-before">${ctxHtml}</div>`
                  : ""
              }
              <div class="message">
                ${messageHtml}
              </div>
            </div>
            <div class="annotator-labels">
              ${annotatorTable}
              ${matchesHtml}
              ${llmDetailsHtml}
            </div>
            ${explanationHtml}
          </article>
        `;
      })
      .join("");

    container.innerHTML = pagination + cards + pagination;

    const prevButtons = document.querySelectorAll("#agreement-prev-page");
    prevButtons.forEach((button) => {
      button.addEventListener("click", () => {
        if (state.page > 1) {
          state.page -= 1;
          renderResults();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      });
    });
    const nextButtons = document.querySelectorAll("#agreement-next-page");
    nextButtons.forEach((button) => {
      button.addEventListener("click", () => {
        if (state.page < totalPages) {
          state.page += 1;
          renderResults();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      });
    });
  }

  function attachHandlers() {
    if (elements.datasetSelect) {
      elements.datasetSelect.addEventListener("change", (event) => {
        if (state.isLoading) {
          event.target.value = state.currentDatasetKey || "";
          return;
        }
        const value = event.target.value || "";
        state.currentDatasetKey = value;
        state.currentAnnotation = "__all__";
        state.currentParticipant = "__all__";
        void loadCurrentDataset();
      });
    }
    if (elements.annotationSelect) {
      elements.annotationSelect.addEventListener("change", (event) => {
        state.currentAnnotation = event.target.value || "__all__";
        updateAnnotationDetails();
        renderMetrics();
        renderResults();
      });
    }
    if (elements.participantSelect) {
      elements.participantSelect.addEventListener("change", (event) => {
        state.currentParticipant = event.target.value || "__all__";
        renderResults();
      });
    }
    if (elements.showAgree) {
      elements.showAgree.addEventListener("change", (event) => {
        state.showAgree = Boolean(event.target.checked);
        state.page = 1;
        renderResults();
      });
    }
    if (elements.showDisagree) {
      elements.showDisagree.addEventListener("change", (event) => {
        state.showDisagree = Boolean(event.target.checked);
        state.page = 1;
        renderResults();
      });
    }
    if (elements.pageSizeSelect) {
      elements.pageSizeSelect.addEventListener("change", (event) => {
        const value = Number(event.target.value);
        state.pageSize = Number.isFinite(value) && value > 0 ? value : 50;
        state.page = 1;
        renderResults();
      });
    }
    if (elements.showContextToggle) {
      elements.showContextToggle.addEventListener("change", (event) => {
        state.showContext = Boolean(event.target.checked);
        state.page = 1;
        renderResults();
      });
    }
    if (elements.showMatchesToggle) {
      elements.showMatchesToggle.addEventListener("change", (event) => {
        state.showMatches = Boolean(event.target.checked);
        state.page = 1;
        renderResults();
      });
    }
    if (elements.showExplanationsToggle) {
      elements.showExplanationsToggle.addEventListener("change", (event) => {
        state.showExplanations = Boolean(event.target.checked);
        state.page = 1;
        renderResults();
      });
    }
    if (elements.scoreCutoffInput) {
      elements.scoreCutoffInput.addEventListener("change", (event) => {
        const rawValue = Number(event.target.value);
        const clamped = Number.isFinite(rawValue)
          ? Math.min(10, Math.max(0, Math.round(rawValue)))
          : state.scoreCutoff;
        state.scoreCutoff = clamped;
        elements.scoreCutoffInput.value = String(clamped);
        state.page = 1;
        void loadCurrentDataset();
      });
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
      if (event.key === "p" || event.key === "P") {
        event.preventDefault();
        state.showContext = !state.showContext;
        if (elements.showContextToggle) {
          elements.showContextToggle.checked = state.showContext;
        }
        state.page = 1;
        renderResults();
      }
    });
  }

  async function initialize() {
    installViewerNav("agreement");
    if (elements.controls && !elements.scoreCutoffInput) {
      elements.scoreCutoffInput = installScoreCutoffControl(
        elements.controls,
        state.scoreCutoff,
        (clamped) => {
          if (state.isLoading) {
            if (elements.scoreCutoffInput) {
              elements.scoreCutoffInput.value = String(state.scoreCutoff);
            }
            return;
          }
          // Snap to the nearest available cutoff for this dataset.
          let target = clamped;
          const available = state.availableScoreCutoffs || [];
          if (available.length) {
            if (!available.includes(target)) {
              target = available.reduce((best, current) => {
                if (best === null) {
                  return current;
                }
                const bestDiff = Math.abs(best - clamped);
                const currentDiff = Math.abs(current - clamped);
                return currentDiff < bestDiff ? current : best;
              }, null);
            }
          }
          if (!Number.isFinite(target)) {
            target = state.scoreCutoff;
          }
          state.scoreCutoff = target;
          if (elements.scoreCutoffInput) {
            elements.scoreCutoffInput.value = String(target);
          }
          state.page = 1;
          void loadCurrentDataset();
        },
      );
    }
    attachHandlers();
    await loadAnnotationSpecs();
    await discoverAgreementDatasets();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      void initialize();
    });
  } else {
    void initialize();
  }
})();
