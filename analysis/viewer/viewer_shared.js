// Shared utilities for analysis viewers (classification and manual annotation).
// This module centralizes HTML escaping, markdown rendering, annotation CSV
// parsing, and context block rendering so multiple UIs stay consistent.

import DOMPurify from "../../node_modules/dompurify/dist/purify.es.mjs";
import snarkdown from "../../node_modules/snarkdown/dist/snarkdown.es.js";

export const REPO_ROOT = "/";
export const ANALYSIS_ROOT = "/analysis/";
export const VIEWER_ROOT = "/analysis/viewer/";
export const ANNOTATION_OUTPUTS_ROOT = "/annotation_outputs/";
export const MANUAL_INPUTS_ROOT = "/manual_annotation_inputs/";
export const MANUAL_LABELS_ROOT = "/manual_annotation_labels/";
export const AGREEMENT_ROOT = "/analysis/agreement/";

// Shared location for the annotations CSV used by all viewers.
export const ANNOTATIONS_CSV_URL = "/api/annotations.csv";

const escapeContainer =
  typeof document !== "undefined" ? document.createElement("span") : null;

export function clampContextDepth(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric) || numeric < 0) {
    return 0;
  }
  if (numeric > 10) {
    return 10;
  }
  return Math.floor(numeric);
}

export function escapeHtml(text) {
  if (!escapeContainer) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
  const value = text === null || text === undefined ? "" : String(text);
  escapeContainer.textContent = value;
  return escapeContainer.innerHTML;
}

function isPureJsonPayload(text) {
  if (text === null || text === undefined) {
    return false;
  }
  const raw = String(text).trim();
  if (!raw) {
    return false;
  }
  if (!raw.startsWith("{") && !raw.startsWith("[")) {
    return false;
  }
  try {
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null;
  } catch {
    return false;
  }
}

export function renderJsonOrMarkdown(text) {
  const source = text === null || text === undefined ? "" : String(text);
  if (isPureJsonPayload(source)) {
    try {
      const parsed = JSON.parse(source);
      const pretty = JSON.stringify(parsed, null, 2);
      return `<pre><code>${escapeHtml(pretty)}</code></pre>`;
    } catch {
      // Fall through to markdown rendering if parsing fails unexpectedly.
    }
  }
  return renderMarkdownToHtml(source);
}

export function renderMessageContent(text, showRaw) {
  const value = text === null || text === undefined ? "" : String(text);
  if (showRaw) {
    return escapeHtml(value).replace(/\n/g, "<br>");
  }
  return renderJsonOrMarkdown(value);
}

export function renderMarkdownToHtml(text) {
  const source = text === null || text === undefined ? "" : String(text);
  const rawHtml = snarkdown(source);
  return DOMPurify.sanitize(rawHtml, {
    ALLOWED_TAGS: [
      "a",
      "em",
      "strong",
      "p",
      "ul",
      "ol",
      "li",
      "code",
      "pre",
      "blockquote",
      "hr",
      "br",
      "del",
    ],
    ALLOWED_ATTR: ["href", "title"],
    ALLOW_DATA_ATTR: false,
  });
}

function normalizeScope(raw) {
  if (!raw) {
    return [];
  }
  return raw
    .replace(/;/g, ",")
    .split(",")
    .map((chunk) => chunk.trim().toLowerCase())
    .filter(Boolean);
}

export function parseAnnotationCsv(text) {
  const parser =
    typeof window !== "undefined" && window.Papa ? window.Papa : null;
  if (!parser || typeof parser.parse !== "function") {
    throw new Error(
      "PapaParse is not available. Ensure frontend dependencies are installed.",
    );
  }

  const result = parser.parse(text, {
    header: true,
    skipEmptyLines: "greedy",
    dynamicTyping: false,
    transformHeader(header) {
      return String(header || "")
        .trim()
        .toLowerCase();
    },
    transform(value) {
      if (value == null) {
        return "";
      }
      return typeof value === "string" ? value.trim() : value;
    },
  });

  if (Array.isArray(result.errors) && result.errors.length) {
    const firstError = result.errors[0];
    const rowInfo =
      typeof firstError.row === "number" ? ` on row ${firstError.row + 1}` : "";
    throw new Error(
      `Unable to parse annotations.csv${rowInfo}: ${firstError.message}`,
    );
  }

  const rows = Array.isArray(result.data) ? result.data : [];

  function splitExamples(raw) {
    if (!raw) return [];
    const textValue = String(raw).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    return textValue
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
  }

  return rows
    .filter((row) => row && Object.values(row).some((value) => value !== ""))
    .map((row) => ({
      id: row.id || "",
      category: row.category || "",
      scope: normalizeScope(row.scope || ""),
      name: row.name || "",
      description: row.description || "",
      original_text: row["original text"] || "",
      positive_examples: splitExamples(row["positive-examples"] || ""),
      negative_examples: splitExamples(row["negative-examples"] || ""),
    }));
}

const ANNOTATION_ID_ALIASES = {
  "theme-awakening-consciousness": "metaphysical-themes",
  "delusion-themes": "metaphysical-themes",
  "claims-unique-understanding": "claims-unique-connection",
};

function normalizeAnnotationIdForLookup(annotationId) {
  const raw = String(annotationId || "").trim();
  if (!raw) {
    return "";
  }
  const lowered = raw.toLowerCase();
  if (lowered.startsWith("assistant-")) {
    return `bot-${raw.slice("assistant-".length)}`;
  }
  if (lowered.startsWith("chatbot-")) {
    return `bot-${raw.slice("chatbot-".length)}`;
  }
  return ANNOTATION_ID_ALIASES[raw] || raw;
}

export function resolveAnnotationSpec(annotationId, annotationSpecs, lookup) {
  const normalized = normalizeAnnotationIdForLookup(annotationId);
  if (!normalized) {
    return null;
  }
  if (lookup && lookup[normalized]) {
    return lookup[normalized];
  }
  const specs = Array.isArray(annotationSpecs) ? annotationSpecs : [];
  const specById =
    lookup || Object.fromEntries(specs.map((spec) => [spec.id, spec]));
  if (specById[normalized]) {
    return specById[normalized];
  }
  if (!normalized.startsWith("user-") && !normalized.startsWith("bot-")) {
    const userSpec = specById[`user-${normalized}`];
    const botSpec = specById[`bot-${normalized}`];
    if (userSpec && botSpec) {
      const combinedDescription =
        `User:\\n\\n${userSpec.description || "(empty)"}\\n\\n` +
        `Assistant:\\n\\n${botSpec.description || "(empty)"}`;
      return {
        id: normalized,
        name: `${userSpec.name || userSpec.id} / ${botSpec.name || botSpec.id}`,
        description: combinedDescription,
        scope: ["user", "assistant"],
        positive_examples: [],
        negative_examples: [],
        combined: {
          user: userSpec,
          assistant: botSpec,
        },
      };
    }
    return userSpec || botSpec || null;
  }
  return null;
}

export function renderContextBlock(label, messages, variant, renderContent) {
  const safeVariant = String(variant || "").toLowerCase();
  const labelChip = `<span class="context-label context-label-${escapeHtml(
    safeVariant,
  )}">${escapeHtml(label)}</span>`;
  const countLabel =
    Array.isArray(messages) && messages.length
      ? `${messages.length} ${messages.length === 1 ? "message" : "messages"}`
      : "No messages";
  const header = `<div class="context-meta">${labelChip} <span class="context-details">${escapeHtml(
    countLabel,
  )}</span></div>`;
  if (!Array.isArray(messages) || !messages.length) {
    return `${header}<div class="context-content muted">No context available.</div>`;
  }
  const itemsHtml = messages
    .map((message, index) => {
      if (!message) {
        return "";
      }
      const detailParts = [];
      const indexValue =
        typeof message.index === "number" ? message.index : index;
      if (typeof indexValue === "number" && Number.isFinite(indexValue)) {
        detailParts.push(`Message ${escapeHtml(String(indexValue + 1))}`);
      }
      const role = message.role ? String(message.role) : "";
      if (role) {
        const roleLabel =
          role.charAt(0).toUpperCase() + role.slice(1).toLowerCase();
        detailParts.push(escapeHtml(roleLabel));
      }
      if (message.timestamp) {
        detailParts.push(escapeHtml(String(message.timestamp)));
      }
      const metaLine = detailParts.length
        ? `<div class="context-item-meta">${detailParts.join(" • ")}</div>`
        : "";
      const contentHtml = renderContent
        ? renderContent(message.content)
        : escapeHtml(String(message.content || ""));
      return `<div class="context-item">${metaLine}<div class="context-item-content">${contentHtml}</div></div>`;
    })
    .join("");
  return `${header}<div class="context-items">${itemsHtml}</div>`;
}

export function renderAnnotationSummaryHtml(spec, options = {}) {
  if (!spec) {
    return "";
  }
  const scope =
    spec.scope && spec.scope.length ? spec.scope.join(", ") : "Any role";
  const description = spec.description || "(empty)";
  const showExamples = Boolean(options.showExamples);
  const renderExampleBlock = (examples, label) => {
    if (!showExamples || !Array.isArray(examples) || !examples.length) {
      return "";
    }
    return `<details><summary>${escapeHtml(label)} (${examples.length})</summary><pre class="pre-box">${escapeHtml(
      examples.join("\n"),
    )}</pre></details>`;
  };

  const posBlock = renderExampleBlock(
    spec.positive_examples,
    "Positive examples",
  );
  const negBlock = renderExampleBlock(
    spec.negative_examples,
    "Negative examples",
  );

  const formatParagraphs = (text) =>
    escapeHtml(text)
      .split(/\n\s*\n/g)
      .map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`)
      .join("");

  let combinedBlock = "";
  if (spec.combined && typeof spec.combined === "object") {
    const userSpec = spec.combined.user;
    const assistantSpec = spec.combined.assistant;
    const userDesc = userSpec ? userSpec.description || "(empty)" : "(empty)";
    const assistantDesc = assistantSpec
      ? assistantSpec.description || "(empty)"
      : "(empty)";
    const userPos = userSpec ? userSpec.positive_examples || [] : [];
    const userNeg = userSpec ? userSpec.negative_examples || [] : [];
    const botPos = assistantSpec ? assistantSpec.positive_examples || [] : [];
    const botNeg = assistantSpec ? assistantSpec.negative_examples || [] : [];
    combinedBlock = `
      <div class="annotation-scope-block">
        <details open>
          <summary>Description (User)</summary>
          ${formatParagraphs(userDesc)}
        </details>
        ${renderExampleBlock(userPos, "User positive examples")}
        ${renderExampleBlock(userNeg, "User negative examples")}
      </div>
      <div class="annotation-scope-block">
        <details open>
          <summary>Description (Assistant)</summary>
          ${formatParagraphs(assistantDesc)}
        </details>
        ${renderExampleBlock(botPos, "Assistant positive examples")}
        ${renderExampleBlock(botNeg, "Assistant negative examples")}
      </div>
    `;
  }

  return `
    <h2>${escapeHtml(spec.name || spec.id)}</h2>
    <p><strong>ID:</strong> ${escapeHtml(spec.id)}</p>
    <p><strong>Scope:</strong> ${escapeHtml(scope)}</p>
    ${
      combinedBlock ||
      `<details open>
        <summary>Description</summary>
        ${formatParagraphs(description)}
      </details>
      ${posBlock}
      ${negBlock}`
    }
  `;
}

// Render a list of span matches as a small highlight block. The caller
// supplies a label prefix (for example, "Matches:" or "gpt-5.1 matches:").
export function renderMatchList(matches, labelPrefix) {
  if (!Array.isArray(matches) || !matches.length) {
    return "";
  }
  const safePrefix = labelPrefix || "Matches:";
  const items = matches
    .filter((value) => typeof value === "string" && value.trim())
    .map(
      (value) => `<div class="match-item">${escapeHtml(String(value))}</div>`,
    )
    .join("");
  if (!items) {
    return "";
  }
  return `<div class="matches"><strong>${escapeHtml(
    safePrefix,
  )}</strong>${items}</div>`;
}

export function renderTextBlock(title, text) {
  const content = typeof text === "string" ? text.trim() : "";
  if (!content) {
    return "";
  }
  return `<details class="llm-detail"><summary>${escapeHtml(
    title,
  )}</summary><pre class="pre-box">${escapeHtml(content)}</pre></details>`;
}

// Fetch directory listing href entries from a python -m http.server index
// page, returning only child paths (files and directories) with relative URLs.
export async function fetchDirectoryEntries(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }
  const text = await response.text();
  const parser = new DOMParser();
  const doc = parser.parseFromString(text, "text/html");
  const anchors = Array.from(doc.querySelectorAll("a"));
  return anchors
    .map((anchor) => anchor.getAttribute("href") || "")
    .filter((href) => href && href !== "../" && !href.startsWith("?"));
}

// Populate the small nav bar at the top of each viewer with links to the
// classification, manual, and agreement pages. The current page is shown as
// plain text; others are rendered as anchors.
export function installViewerNav(current) {
  const nav = document.querySelector(".viewer-nav");
  if (!nav) {
    return;
  }
  const entries = [
    {
      key: "classification",
      href: "classification_viewer.html",
      label: "Classification",
    },
    { key: "manual", href: "manual_annotator.html", label: "Manual" },
    {
      key: "agreement",
      href: "annotation_agreement_viewer.html",
      label: "Agreement",
    },
  ];
  const currentKey = String(current || "").toLowerCase();
  const parts = entries.map((entry) => {
    const label = escapeHtml(entry.label);
    if (entry.key === currentKey) {
      return `<span>${label}</span>`;
    }
    return `<a href="${escapeHtml(entry.href)}">${label}</a>`;
  });
  nav.innerHTML = parts.join(" | ");
}

export function applyDatasetLabelOrFallback(item, fallbackKey) {
  const rawLabel =
    item && typeof item.label === "string" ? item.label.trim() : "";
  if (rawLabel) {
    return rawLabel;
  }
  return String(fallbackKey || "");
}

export function buildDataFileUrl(relativePath) {
  const trimmed = String(relativePath || "").replace(/^\/+/, "");
  if (!trimmed) {
    return "";
  }
  const encoded = trimmed.replace(/%/g, "%25");
  return `/${encoded}`;
}

// Install a shared "Score cutoff" control in the provided controls container.
// The onChange callback receives a clamped integer cutoff (0–10).
export function installScoreCutoffControl(
  controlsElement,
  initialValue,
  onChange,
) {
  if (!controlsElement || typeof onChange !== "function") {
    return null;
  }
  const group = document.createElement("div");
  group.className = "control-group";
  group.innerHTML = `
      <label for="score-cutoff-input">Score cutoff</label>
      <input id="score-cutoff-input" type="number" min="0" max="10" step="1" value="">
      <span class="control-note">Messages with integer score \u2265 cutoff are treated as matches when scores are available.</span>
    `;
  controlsElement.appendChild(group);
  const input = group.querySelector("#score-cutoff-input");
  if (!input) {
    return null;
  }
  input.value = String(initialValue);
  input.addEventListener("change", (event) => {
    const rawValue = Number(event.target.value);
    const clamped = Number.isFinite(rawValue)
      ? Math.min(10, Math.max(0, Math.round(rawValue)))
      : initialValue;
    if (String(clamped) !== event.target.value) {
      event.target.value = String(clamped);
    }
    onChange(clamped);
  });
  return input;
}

// Populate a <select> element with annotation options grouped by category,
// using annotations.csv specs plus any dataset-local lookup. Returns the new
// selected value while preserving a "none" placeholder when provided.
export function populateGroupedAnnotationOptions(
  selectElement,
  annotationIds,
  annotationSpecs,
  datasetLookup,
  currentValue,
  noneValue,
) {
  if (!selectElement) {
    return currentValue;
  }
  const select = selectElement;
  const ids = Array.from(new Set((annotationIds || []).filter((id) => !!id)));
  while (select.options.length > 1 && noneValue !== undefined) {
    select.remove(1);
  }
  if (noneValue === undefined) {
    while (select.options.length) {
      select.remove(0);
    }
  }

  if (!ids.length) {
    if (noneValue !== undefined) {
      select.value = noneValue;
    }
    return noneValue !== undefined ? noneValue : "";
  }

  const byId = new Map();
  if (Array.isArray(annotationSpecs) && annotationSpecs.length) {
    annotationSpecs.forEach((spec) => {
      if (!spec || !spec.id) return;
      byId.set(spec.id, {
        id: spec.id,
        name: spec.name || "",
        category: spec.category || "Uncategorized",
      });
    });
  }
  const lookup = datasetLookup || {};
  Object.keys(lookup).forEach((id) => {
    if (!byId.has(id)) {
      const entry = lookup[id] || {};
      byId.set(id, {
        id,
        name: entry.name || "",
        category: entry.category || "Uncategorized",
      });
    }
  });

  const groups = new Map();
  ids.forEach((id) => {
    const data = byId.get(id) || { id, name: "", category: "Uncategorized" };
    const cat = data.category || "Uncategorized";
    if (!groups.has(cat)) groups.set(cat, []);
    groups.get(cat).push(data);
  });

  const collator = new Intl.Collator("en", {
    numeric: true,
    sensitivity: "base",
  });
  const categories = Array.from(groups.keys()).sort((a, b) =>
    collator.compare(a, b),
  );

  categories.forEach((cat) => {
    const opts = groups.get(cat) || [];
    opts.sort((a, b) => collator.compare(a.id, b.id));
    const og = document.createElement("optgroup");
    og.label = cat;
    opts.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent =
        entry.name && entry.name !== entry.id
          ? `${entry.id}: ${entry.name}`
          : entry.id;
      og.appendChild(option);
    });
    select.appendChild(og);
  });

  let nextValue = currentValue;
  if (!nextValue || !ids.includes(nextValue)) {
    const firstCat = categories[0];
    const firstEntry = firstCat ? (groups.get(firstCat) || [])[0] : null;
    nextValue = firstEntry ? firstEntry.id : ids[0];
  }
  select.value = nextValue;
  return nextValue;
}
