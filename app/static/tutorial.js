/* Guided comparison tour. No audio or call processing is changed by the tour. */
(() => {
  const storageKey = 'calmline.comparison-tour.v1';
  const steps = [
    {
      target: '.side:not(.protected) .side-head',
      title: 'One call, two experiences',
      text: 'The left side shows the original caller. The right side shows the calmer voice, transformed text, and complaint details an agent receives.'
    },
    {
      target: '.input-area',
      title: 'Choose how to try it',
      text: 'Select a sample and press Play call, speak into your microphone, or type a message. Start with a sample to hear the before-and-after difference. Microphone input is in English; use headphones.'
    },
    {
      target: '.side:not(.protected) .audio-control',
      title: 'Listen to the original caller',
      text: 'The original voice starts at 15% volume. Raise the slider to hear the emotion and abusive language, or switch it OFF. Switching it back ON restores your chosen volume.'
    },
    {
      target: '.protected .audio-control',
      title: 'Hear the CalmLine voice',
      text: 'The calmer voice starts at 100%. Adjust it independently or switch it OFF to compare. Both voices play together by default, with a processing delay. This is a judge’s comparison view; agents hear only CalmLine.'
    },
    {
      target: '#original-list',
      title: 'Follow the original words',
      text: 'The transcript appears as each part is processed. Highlighted phrases show detected abuse. Match the sentence numbers on the left and right to see what changed.'
    },
    {
      target: '#calm-list',
      title: 'Compare the same sentence',
      text: 'See the calmer message, unchanged speech, or abuse that was withheld. Once a sample call finishes, use the matching Replay buttons to hear the original and calmer versions of the same sentence. Typed messages have no original audio.'
    },
    {
      target: '.facts',
      title: 'Watch the complaint take shape',
      text: 'The complaint type, location, timing, and other extracted details build up here. Safety alerts appear separately when a threat is detected. You’re ready—play a sample, and reopen this tour anytime with How to use.'
    }
  ];

  const layer = document.createElement('div');
  layer.className = 'tour-layer';
  layer.hidden = true;
  layer.innerHTML = `
    <div class="tour-backdrop"></div>
    <div class="tour-highlight" aria-hidden="true"></div>
    <section class="tour-panel" role="dialog" aria-modal="true"
      aria-labelledby="tour-title" aria-describedby="tour-description" tabindex="-1">
      <div class="tour-top">
        <span class="tour-count" aria-live="polite"></span>
        <button class="tour-close" aria-label="Close tutorial">×</button>
      </div>
      <h2 id="tour-title"></h2>
      <p id="tour-description"></p>
      <div class="tour-progress" aria-hidden="true"></div>
      <div class="tour-actions">
        <button class="link tour-skip">Skip tour</button>
        <button class="btn tour-back">Back</button>
        <button class="btn primary tour-next">Next</button>
      </div>
    </section>`;
  document.body.append(layer);

  const panel = layer.querySelector('.tour-panel');
  const highlight = layer.querySelector('.tour-highlight');
  const next = layer.querySelector('.tour-next');
  const back = layer.querySelector('.tour-back');
  const launch = document.getElementById('tour-launch');
  let current = 0;
  let previousFocus;
  let previousScroll;
  let inertElements = [];
  let frame = null;

  function position() {
    frame = null;
    if (layer.hidden) return;
    const target = document.querySelector(steps[current].target);
    if (!target) return;
    const rect = target.getBoundingClientRect();
    const width = window.innerWidth;
    const height = window.innerHeight;
    const left = Math.max(8, rect.left - 6);
    const top = Math.max(8, rect.top - 6);
    const right = Math.min(width - 8, rect.right + 6);
    const bottom = Math.min(height - 8, rect.bottom + 6);
    Object.assign(highlight.style, {
      left: `${left}px`, top: `${top}px`,
      width: `${Math.max(0, right - left)}px`,
      height: `${Math.max(0, bottom - top)}px`
    });

    const panelWidth = panel.offsetWidth;
    const panelHeight = panel.offsetHeight;
    const center = (left + right) / 2;
    let x = Math.max(16, Math.min(center - panelWidth / 2, width - panelWidth - 16));
    let y;
    let side;
    if (bottom + 16 + panelHeight <= height - 16) {
      y = bottom + 16;
      side = 'below';
    } else if (top - 16 - panelHeight >= 16) {
      y = top - 16 - panelHeight;
      side = 'above';
    } else if (width - right - 20 >= panelWidth) {
      x = right + 20;
      y = Math.max(16, Math.min(top, height - panelHeight - 16));
      side = 'beside';
    } else if (left - 20 >= panelWidth) {
      x = left - panelWidth - 20;
      y = Math.max(16, Math.min(top, height - panelHeight - 16));
      side = 'beside';
    } else {
      y = Math.max(16, height - panelHeight - 16);
      side = 'floating';
    }
    Object.assign(panel.style, { left: `${x}px`, top: `${y}px` });
    panel.dataset.side = side;
    panel.style.setProperty('--arrow-x', `${Math.max(x + 22, Math.min(center - 6, x + panelWidth - 34))}px`);
    panel.style.setProperty('--panel-top', `${y}px`);
    panel.style.setProperty('--panel-bottom', `${y + panelHeight}px`);
  }

  function reposition() {
    if (!layer.hidden && frame === null) frame = requestAnimationFrame(position);
  }

  function showStep() {
    const step = steps[current];
    layer.querySelector('.tour-count').textContent = `QUICK TOUR · ${current + 1} OF ${steps.length}`;
    layer.querySelector('#tour-title').textContent = step.title;
    layer.querySelector('#tour-description').textContent = step.text;
    layer.querySelector('.tour-progress').innerHTML = steps.map((_, index) =>
      `<span class="${index === current ? 'current' : index < current ? 'past' : ''}"></span>`
    ).join('');
    back.hidden = current === 0;
    next.textContent = current === steps.length - 1 ? 'Try it now' : 'Next';
    document.querySelector(step.target)?.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
    position();
    // Announce each new step and keep keyboard navigation inside the tutorial.
    panel.focus({ preventScroll: true });
  }

  function open() {
    if (!layer.hidden) return;
    previousFocus = document.activeElement;
    previousScroll = { x: window.scrollX, y: window.scrollY };
    current = 0;
    layer.hidden = false;
    inertElements = [...document.body.children]
      .filter(element => element !== layer && !['SCRIPT', 'STYLE'].includes(element.tagName))
      .map(element => ({ element, inert: element.inert }));
    for (const { element } of inertElements) element.inert = true;
    showStep();
  }

  function close() {
    if (layer.hidden) return;
    layer.hidden = true;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    for (const { element, inert } of inertElements) element.inert = inert;
    inertElements = [];
    try { localStorage.setItem(storageKey, 'seen'); } catch { /* Storage can be unavailable. */ }
    window.scrollTo(previousScroll.x, previousScroll.y);
    (previousFocus?.isConnected ? previousFocus : launch).focus({ preventScroll: true });
  }

  launch.addEventListener('click', open);
  layer.querySelector('.tour-close').addEventListener('click', close);
  layer.querySelector('.tour-skip').addEventListener('click', close);
  back.addEventListener('click', () => { if (current > 0) { current--; showStep(); } });
  next.addEventListener('click', () => {
    if (current === steps.length - 1) close();
    else { current++; showStep(); }
  });
  layer.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); close(); }
    else if (event.key === 'ArrowRight') { event.preventDefault(); next.click(); }
    else if (event.key === 'ArrowLeft' && current > 0) { event.preventDefault(); back.click(); }
    else if (event.key === 'Tab') {
      const controls = [...panel.querySelectorAll('button')].filter(button => !button.hidden && !button.disabled);
      const index = controls.indexOf(document.activeElement);
      const destination = event.shiftKey
        ? (index <= 0 ? controls.length - 1 : index - 1)
        : (index + 1) % controls.length;
      event.preventDefault();
      controls[destination].focus();
    }
  });
  window.addEventListener('resize', reposition);
  window.addEventListener('scroll', reposition, true);
  let seen = false;
  try { seen = localStorage.getItem(storageKey) === 'seen'; } catch { /* Show the tour without storage. */ }
  if (!seen) setTimeout(() => { if (!busy && cards.size === 0) open(); }, 500);
})();
