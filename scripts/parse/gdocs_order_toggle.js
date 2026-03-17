"use strict";

/**
 * Convert an integer to a two-digit chat label string.
 *
 * @param {number} num Chat index (0-based).
 * @returns {string} Padded label such as "01" or "12".
 */
function padChatIndex(num) {
  const value = String(num);
  return value.length >= 2 ? value : "0" + value;
}

/**
 * Update the state of previous/next navigation buttons.
 *
 * @param {object} state Shared interaction state.
 */
function updateNavState(state) {
  const lastIndex = state.currentOrder.length - 1;
  state.currentOrder.forEach(function updateNav(originalIndex, position) {
    const section = state.sections[originalIndex];
    if (!section) {
      return;
    }
    section.setAttribute("data-position", String(position));
    const prevBtn = section.querySelector(".chat-nav-prev");
    const nextBtn = section.querySelector(".chat-nav-next");
    if (prevBtn) {
      prevBtn.disabled = position === 0;
    }
    if (nextBtn) {
      nextBtn.disabled = position === lastIndex;
    }
  });
}

/**
 * Render chats and TOC entries according to the provided order.
 *
 * @param {object} state Shared interaction state.
 * @param {number[]} order Array mapping new positions to original indices.
 */
function renderOrder(state, order) {
  state.currentOrder = order.slice();
  state.currentOrder.forEach(function placeChat(originalIndex, position) {
    const section = state.sections[originalIndex];
    const tocItem = state.tocItems[originalIndex];
    if (!section) {
      return;
    }

    const anchorId = "chat-" + (position + 1);
    state.container.appendChild(section);
    section.setAttribute("data-position", String(position));

    const heading = section.querySelector(".chat-heading");
    if (heading) {
      heading.id = anchorId;
      heading.setAttribute("data-position", String(position));
      const label = heading.querySelector(".chat-label");
      if (label) {
        label.textContent = "Chat " + padChatIndex(position + 1);
      }
    }

    if (state.toc && tocItem) {
      state.toc.appendChild(tocItem);
      const link = tocItem.querySelector("a");
      if (link) {
        link.setAttribute("href", "#" + anchorId);
        const label = tocItem.querySelector(".chat-label");
        if (label) {
          label.textContent = "Chat " + padChatIndex(position + 1);
        }
      }
    }
  });
  updateNavState(state);
  state.messages = Array.from(state.container.querySelectorAll(".message"));
  applyRoleFilter(state);
}

/**
 * Toggle the visibility of a chat section.
 *
 * @param {HTMLElement} section Chat section element.
 * @param {HTMLElement} control Button that triggered the toggle.
 */
function toggleSection(section, control) {
  const body = section.querySelector(".chat-body");
  if (!body) {
    return;
  }
  const isCollapsed = body.classList.toggle("is-collapsed");
  body.setAttribute("data-collapsed", isCollapsed ? "true" : "false");
  control.textContent = isCollapsed ? "Show Chat" : "Hide Chat";
  control.setAttribute("data-state", isCollapsed ? "collapsed" : "expanded");
}

/**
 * Jump to the chat relative to the current one.
 *
 * @param {object} state Shared interaction state.
 * @param {HTMLElement} section Current chat section.
 * @param {number} direction +1 for next, -1 for previous.
 */
function jumpToRelative(state, section, direction) {
  const position = Number(section.getAttribute("data-position") || "0");
  const targetPosition = position + direction;
  if (targetPosition < 0 || targetPosition >= state.currentOrder.length) {
    return;
  }
  const targetSection = state.sections[state.currentOrder[targetPosition]];
  if (!targetSection) {
    return;
  }
  setSectionExpanded(targetSection, true);
  targetSection.scrollIntoView({ behavior: "smooth", block: "start" });
  const heading = targetSection.querySelector(".chat-heading");
  if (heading instanceof HTMLElement) {
    heading.focus({ preventScroll: true });
  }
}

/**
 * Expand or collapse a specific chat section deterministically.
 *
 * @param {HTMLElement} section Chat section element.
 * @param {boolean} expand True to expand, false to collapse.
 */
function setSectionExpanded(section, expand) {
  const body = section.querySelector(".chat-body");
  const toggle = section.querySelector(".chat-toggle");
  if (!body) {
    return;
  }
  if (expand) {
    body.classList.remove("is-collapsed");
    body.setAttribute("data-collapsed", "false");
    if (toggle) {
      toggle.textContent = "Hide Chat";
      toggle.setAttribute("data-state", "expanded");
    }
  } else {
    body.classList.add("is-collapsed");
    body.setAttribute("data-collapsed", "true");
    if (toggle) {
      toggle.textContent = "Show Chat";
      toggle.setAttribute("data-state", "collapsed");
    }
  }
}

/**
 * Expand or collapse all chat sections.
 *
 * @param {object} state Shared interaction state.
 * @param {boolean} expand True to expand all, false to collapse all.
 */
function setAllSectionsExpanded(state, expand) {
  if (!Array.isArray(state.sections)) {
    return;
  }
  state.sections.forEach(function applySet(section) {
    setSectionExpanded(section, expand);
  });
}

/**
 * Update the Expand/Collapse All button label to reflect current state.
 *
 * @param {object} state Shared interaction state.
 */
function updateExpandAllButton(state) {
  const btn = state.expandAllButton;
  if (!btn || !Array.isArray(state.sections) || state.sections.length === 0) {
    return;
  }
  const allExpanded = state.sections.every(function isExpanded(section) {
    const body = section.querySelector(".chat-body");
    return body && !body.classList.contains("is-collapsed");
  });
  if (allExpanded) {
    btn.textContent = "Collapse All Chats";
    btn.setAttribute("data-state", "expanded");
  } else {
    btn.textContent = "Expand All Chats";
    btn.setAttribute("data-state", "collapsed");
  }
}

/**
 * Apply the role filter to all messages.
 *
 * @param {object} state Shared interaction state.
 */
function applyRoleFilter(state) {
  if (!state.messages || state.messages.length === 0) {
    return;
  }
  state.messages.forEach(function filterMessage(message) {
    const messageRole = message.getAttribute("data-role") || "unknown";
    const roleVisibility = state.roleVisibility || {};
    const manualState = message.getAttribute("data-manual-state") || "visible";
    const overrideActive =
      message.getAttribute("data-filter-override") === "true";
    const toggle = message.querySelector(".message-toggle");
    const indicator = message.querySelector(".filter-indicator");

    const shouldFilter =
      Object.prototype.hasOwnProperty.call(roleVisibility, messageRole) &&
      roleVisibility[messageRole] === false;

    if (shouldFilter && !overrideActive) {
      message.classList.add("is-filtered");
      message.classList.remove("is-filter-overridden");
    } else if (shouldFilter && overrideActive) {
      message.classList.add("is-filter-overridden");
      message.classList.remove("is-filtered");
    } else {
      message.classList.remove("is-filtered");
      message.classList.remove("is-filter-overridden");
      if (!shouldFilter) {
        message.removeAttribute("data-filter-override");
      }
    }

    if (manualState === "hidden") {
      message.classList.add("is-manually-hidden");
    } else {
      message.classList.remove("is-manually-hidden");
    }

    if (indicator) {
      if (shouldFilter && !overrideActive) {
        indicator.removeAttribute("hidden");
        indicator.textContent = "Filtered";
      } else if (shouldFilter && overrideActive) {
        indicator.removeAttribute("hidden");
        indicator.textContent = "Filtered (shown)";
      } else {
        indicator.setAttribute("hidden", "hidden");
        indicator.textContent = "";
      }
    }

    const finalVisible =
      manualState !== "hidden" && (!shouldFilter || overrideActive);
    if (toggle) {
      toggle.textContent = finalVisible ? "Hide message" : "Show message";
      toggle.setAttribute("data-state", finalVisible ? "visible" : "hidden");
    }
  });
  updateFloatingButton(state);
}

/**
 * Return the chat section that should be considered active in the viewport.
 *
 * @param {object} state Shared interaction state.
 * @returns {HTMLElement | null} Active chat section, if any.
 */
function findActiveChatSection(state) {
  if (
    !Array.isArray(state.currentOrder) ||
    state.currentOrder.length === 0 ||
    !Array.isArray(state.sections)
  ) {
    return null;
  }

  const anchorOffset = Math.min(window.innerHeight * 0.25, 200);
  let lastVisible = null;
  let firstUpcoming = null;

  state.currentOrder.forEach(function pickSection(originalIndex) {
    const section = state.sections[originalIndex];
    if (!section || !section.isConnected) {
      return;
    }
    const rect = section.getBoundingClientRect();
    if (rect.bottom <= 0) {
      lastVisible = section;
      return;
    }
    if (rect.top <= anchorOffset) {
      lastVisible = section;
      return;
    }
    if (!firstUpcoming) {
      firstUpcoming = section;
    }
  });

  if (lastVisible) {
    return lastVisible;
  }
  if (firstUpcoming) {
    return firstUpcoming;
  }

  const firstSection = state.sections[state.currentOrder[0]];
  return firstSection && firstSection.isConnected ? firstSection : null;
}

/**
 * Update the floating navigation button to reflect the active chat.
 *
 * @param {object} state Shared interaction state.
 */
function updateFloatingButton(state) {
  const button = state.scrollButton;
  if (!button) {
    return;
  }

  const hasSections =
    Array.isArray(state.sections) && state.sections.length > 0;

  if (!hasSections) {
    button.textContent = "Back to top";
    button.setAttribute("data-target-type", "container");
    button.removeAttribute("data-target-id");
    button.removeAttribute("hidden");
    return;
  }

  const activeSection = findActiveChatSection(state);
  if (!activeSection) {
    button.setAttribute("hidden", "hidden");
    return;
  }

  const heading =
    activeSection.querySelector(".chat-heading") ||
    activeSection.firstElementChild;
  if (!(heading instanceof HTMLElement) || !heading.id) {
    button.setAttribute("hidden", "hidden");
    return;
  }

  const labelElement = heading.querySelector(".chat-label");
  const labelText =
    labelElement && labelElement.textContent
      ? labelElement.textContent.trim()
      : "Chat start";

  button.textContent = "Jump to " + labelText;
  button.setAttribute("data-target-type", "heading");
  button.setAttribute("data-target-id", heading.id);
  button.removeAttribute("hidden");
}

/**
 * Schedule a floating button update on the next animation frame.
 *
 * @param {object} state Shared interaction state.
 */
function scheduleFloatingButtonUpdate(state) {
  if (state.updateScheduled) {
    return;
  }
  state.updateScheduled = true;
  window.requestAnimationFrame(function handleScheduledUpdate() {
    state.updateScheduled = false;
    updateFloatingButton(state);
  });
}

/**
 * Handle clicks on the floating navigation button.
 *
 * @param {object} state Shared interaction state.
 */
function handleScrollButtonClick(state) {
  const button = state.scrollButton;
  if (!button) {
    return;
  }

  const targetType = button.getAttribute("data-target-type") || "heading";
  if (targetType === "container") {
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }

  const targetId = button.getAttribute("data-target-id") || "";
  if (!targetId) {
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }

  const targetElement = document.getElementById(targetId);
  if (targetElement) {
    targetElement.scrollIntoView({ behavior: "smooth", block: "start" });
    if (targetElement instanceof HTMLElement) {
      targetElement.focus({ preventScroll: true });
    }
  }
}

/**
 * Toggle visibility for an individual message, respecting manual and filter states.
 *
 * @param {object} state Shared interaction state.
 * @param {HTMLElement} message Message container element.
 * @param {HTMLElement} control Button that triggered the toggle.
 */
function toggleMessageVisibility(state, message, control) {
  const manualState = message.getAttribute("data-manual-state") || "visible";
  const isCurrentlyVisible = control.getAttribute("data-state") !== "hidden";
  const isFiltered = message.classList.contains("is-filtered");
  const overrideActive =
    message.getAttribute("data-filter-override") === "true";

  if (!isCurrentlyVisible) {
    if (manualState === "hidden") {
      message.setAttribute("data-manual-state", "visible");
      if (isFiltered && !overrideActive) {
        message.setAttribute("data-filter-override", "true");
      } else if (!isFiltered) {
        message.removeAttribute("data-filter-override");
      }
    } else if (isFiltered && !overrideActive) {
      message.setAttribute("data-filter-override", "true");
    } else {
      message.removeAttribute("data-filter-override");
    }
    applyRoleFilter(state);
    return;
  }

  if (overrideActive) {
    message.removeAttribute("data-filter-override");
  }
  message.setAttribute("data-manual-state", "hidden");
  applyRoleFilter(state);
}

/**
 * Handle clicks on the order toggle button.
 *
 * @param {object} state Shared interaction state.
 */
function handleOrderToggle(state) {
  state.reversed = !state.reversed;
  const order = state.reversed
    ? state.defaultOrder.slice().reverse()
    : state.defaultOrder.slice();
  renderOrder(state, order);
  if (state.orderButton) {
    state.orderButton.textContent = state.reversed
      ? "Show Newest First"
      : "Show Oldest First";
  }
  applyRoleFilter(state);
}

/**
 * Handle click events inside the chat container.
 *
 * @param {object} state Shared interaction state.
 * @param {MouseEvent} event Click event.
 */
function handleContainerClick(state, event) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  if (target.classList.contains("message-toggle")) {
    const message = target.closest(".message");
    if (!message) {
      return;
    }
    toggleMessageVisibility(state, message, target);
    return;
  }

  if (target.classList.contains("chat-toggle")) {
    const section = target.closest("section.chat");
    if (section) {
      toggleSection(section, target);
    }
    return;
  }

  if (target.classList.contains("chat-nav")) {
    const section = target.closest("section.chat");
    if (!section) {
      return;
    }
    const direction = Number(target.getAttribute("data-dir") || "0");
    if (!direction) {
      return;
    }
    jumpToRelative(state, section, direction);
  }
}

/**
 * Initialize chat interaction handlers for Google Docs exports.
 */
function initGdocsInteractions() {
  const container = document.getElementById("chat-container");
  if (!container) {
    return;
  }
  const toc = document.getElementById("toc-list");

  const orderButton = document.getElementById("order-toggle");
  const expandAllButton = document.getElementById("expand-collapse-all");
  const sections = Array.from(container.querySelectorAll("section.chat"));
  const tocItems = toc ? Array.from(toc.querySelectorAll("li.toc-item")) : [];
  if (toc && sections.length !== tocItems.length) {
    return;
  }
  const roleRadios = {
    user: Array.from(
      document.querySelectorAll('input[name="role-user"][type="radio"]'),
    ),
    assistant: Array.from(
      document.querySelectorAll('input[name="role-assistant"][type="radio"]'),
    ),
    tool: Array.from(
      document.querySelectorAll('input[name="role-tool"][type="radio"]'),
    ),
  };
  const scrollButton = document.getElementById("scroll-to-chat-start");

  const defaultOrder = sections.map(function mapOrder(_section, index) {
    return index;
  });

  const state = {
    container,
    toc,
    orderButton,
    expandAllButton,
    sections,
    tocItems,
    defaultOrder,
    currentOrder: defaultOrder.slice(),
    reversed: false,
    scrollButton,
    updateScheduled: false,
    roleRadios,
    roleVisibility: {
      user: true,
      assistant: true,
      tool: true,
    },
    messages: Array.from(container.querySelectorAll(".message")),
  };

  renderOrder(state, state.defaultOrder);
  updateExpandAllButton(state);

  if (state.orderButton) {
    state.orderButton.addEventListener("click", function onOrderClick() {
      handleOrderToggle(state);
    });
  }

  if (state.expandAllButton) {
    state.expandAllButton.addEventListener(
      "click",
      function onExpandAllClick() {
        const current =
          state.expandAllButton.getAttribute("data-state") || "collapsed";
        const expand = current !== "expanded";
        setAllSectionsExpanded(state, expand);
        updateExpandAllButton(state);
      },
    );
  }

  state.container.addEventListener("click", function onContainerClick(event) {
    handleContainerClick(state, event);
  });

  if (state.scrollButton) {
    state.scrollButton.addEventListener(
      "click",
      function onScrollButtonClick() {
        handleScrollButtonClick(state);
      },
    );
  }

  window.addEventListener("scroll", function onWindowScroll() {
    scheduleFloatingButtonUpdate(state);
  });
  window.addEventListener("resize", function onWindowResize() {
    scheduleFloatingButtonUpdate(state);
  });

  // Ensure the chat referenced by the URL hash is expanded on load and when
  // the hash changes. This helps when deep-linking into large documents where
  // all chats start collapsed by default.
  function expandChatFromHash() {
    const raw = window.location.hash || "";
    const id = raw.startsWith("#") ? raw.slice(1) : raw;
    if (!id) {
      return;
    }
    const heading = document.getElementById(id);
    if (!heading) {
      return;
    }
    const section = heading.closest("section.chat");
    if (!section) {
      return;
    }
    setSectionExpanded(section, true);
  }

  // Expand on initial load and on hash changes.
  function expandFromHashAndUpdate() {
    expandChatFromHash();
    updateExpandAllButton(state);
  }
  expandFromHashAndUpdate();
  window.addEventListener("hashchange", expandFromHashAndUpdate);

  Object.entries(state.roleRadios).forEach(function registerRoleRadios(entry) {
    const role = entry[0];
    const radios = entry[1];
    if (!Array.isArray(radios) || radios.length === 0) {
      delete state.roleVisibility[role];
      return;
    }
    const selected = radios.find(function isChecked(radio) {
      return radio.checked;
    });
    state.roleVisibility[role] =
      selected && selected.value === "hide" ? false : true;
    radios.forEach(function attachRadio(radio) {
      radio.addEventListener("change", function onRoleRadioChange(event) {
        if (!(event.target instanceof HTMLInputElement)) {
          return;
        }
        state.roleVisibility[role] =
          event.target.value === "hide" ? false : true;
        applyRoleFilter(state);
      });
    });
  });

  scheduleFloatingButtonUpdate(state);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initGdocsInteractions);
} else {
  initGdocsInteractions();
}

window.initGdocsInteractions = initGdocsInteractions;
